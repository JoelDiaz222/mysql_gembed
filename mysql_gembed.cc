/* Copyright (c) 2025, Joel Díaz
 *
 * MySQL Component for generating embeddings using the Gembed Rust static library
 * Inspired by lefred's vector_operations component
 * Requires MySQL 9.0+ for native VECTOR datatype support
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License, version 2,
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */

#include <mysql/components/component_implementation.h>
#include <mysql/components/services/mysql_string.h>
#include <mysql/components/services/udf_metadata.h>
#include <mysql/components/services/udf_registration.h>
#include <mysql/components/services/log_builtins.h>
#include <mysqld_error.h>
#include <cstring>
#include "mysql_gembed.h"

#define MYSQL_ERRMSG_SIZE 512

REQUIRES_SERVICE_PLACEHOLDER(mysql_udf_metadata);
REQUIRES_SERVICE_PLACEHOLDER(udf_registration);
REQUIRES_SERVICE_PLACEHOLDER(log_builtins);
REQUIRES_SERVICE_PLACEHOLDER(log_builtins_string);

BEGIN_COMPONENT_PROVIDES(component_mysql_gembed)
END_COMPONENT_PROVIDES();

BEGIN_COMPONENT_REQUIRES(component_mysql_gembed)
  REQUIRES_SERVICE(mysql_udf_metadata),
  REQUIRES_SERVICE(udf_registration),
  REQUIRES_SERVICE(log_builtins),
  REQUIRES_SERVICE(log_builtins_string),
END_COMPONENT_REQUIRES();

/* Component metadata */
BEGIN_COMPONENT_METADATA(component_mysql_gembed)
  METADATA("mysql.author", "Joel Díaz"),
  METADATA("mysql.license", "GPL"),
  METADATA("mysql.dev", "Joel Díaz"),
END_COMPONENT_METADATA();

/* Forward declarations */
static bool generate_embedding_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
static void generate_embedding_deinit(UDF_INIT *initid);
static char *generate_embedding(UDF_INIT *initid, UDF_ARGS *args,
                                char *result, unsigned long *length,
                                unsigned char *is_null, unsigned char *error);

static bool generate_embeddings_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
static void generate_embeddings_deinit(UDF_INIT *initid);
static char *generate_embeddings(UDF_INIT *initid, UDF_ARGS *args,
                                       char *result, unsigned long *length,
                                       unsigned char *is_null, unsigned char *error);

static void log_message(int severity, const char *msg) {
    if (mysql_service_log_builtins && mysql_service_log_builtins->message) {
        mysql_service_log_builtins->message(severity, ER_LOG_PRINTF_MSG,
                                            "component_mysql_gembed: %s", msg);
    }
}

/* UDF: GENERATE_EMBEDDING(method, model, text) -> VECTOR */
static bool generate_embedding_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 3) {
        snprintf(message, MYSQL_ERRMSG_SIZE,
                "GENERATE_EMBEDDING requires 3 arguments: method, model, text");
        return true;
    }

    if (args->arg_type[0] != STRING_RESULT ||
        args->arg_type[1] != STRING_RESULT ||
        args->arg_type[2] != STRING_RESULT) {
        snprintf(message, MYSQL_ERRMSG_SIZE, "All arguments must be strings");
        return true;
    }

    // Set return type as VECTOR (binary data)
    initid->maybe_null = true;
    // Max vector size
    initid->max_length = 65535;
    // Allocate context for storing the vector result
    initid->ptr = nullptr;

    return false;
}

static void generate_embedding_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        delete[] initid->ptr;
        initid->ptr = nullptr;
    }
}

static char *generate_embedding(UDF_INIT *initid, UDF_ARGS *args,
                                char * /*result*/, unsigned long *length,
                                unsigned char *is_null, unsigned char *error) {
    const char *method = args->args[0];
    const char *model = args->args[1];
    const char *text = args->args[2];

    if (!method || !model || !text) {
        *is_null = 1;
        return nullptr;
    }

    // Validate method
    int method_id = validate_embedding_method(method);
    if (method_id < 0) {
        *error = 1;
        log_message(ERROR_LEVEL, "Invalid embedding method");
        return nullptr;
    }

    // Validate model
    int model_id = validate_embedding_model(method_id, model, INPUT_TYPE_TEXT);
    if (model_id < 0) {
        *error = 1;
        log_message(ERROR_LEVEL, "Invalid or unsupported model");
        return nullptr;
    }

    // Prepare input
    StringSlice input;
    input.ptr = text;
    input.len = args->lengths[2];

    // Generate embedding
    EmbeddingBatch batch;
    int err = generate_embeddings_from_texts(method_id, model_id, &input, 1, &batch);

    if (err != 0 || batch.n_vectors != 1) {
        free_embedding_batch(&batch);
        *error = 1;
        log_message(ERROR_LEVEL, "Embedding generation failed");
        return nullptr;
    }

    // MySQL 9.0 VECTOR format: binary representation of floats
    // Format: dimension count (4 bytes) + float array
    size_t vector_size = sizeof(uint32_t) + (batch.dim * sizeof(float));
    char *vector_data = new char[vector_size];

    // Write dimension
    *reinterpret_cast<uint32_t*>(vector_data) = static_cast<uint32_t>(batch.dim);

    // Write float data
    memcpy(vector_data + sizeof(uint32_t), batch.data, batch.dim * sizeof(float));

    free_embedding_batch(&batch);

    // Store in UDF context
    if (initid->ptr) {
        delete[] initid->ptr;
    }
    initid->ptr = vector_data;
    *length = vector_size;

    return vector_data;
}

/* UDF: GENERATE_EMBEDDINGS(method, model, JSON_ARRAY(texts)) -> JSON_ARRAY(vectors) */
static bool generate_embeddings_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 3) {
        snprintf(message, MYSQL_ERRMSG_SIZE,
                "GENERATE_EMBEDDINGS requires 3 arguments: method, model, texts_json");
        return true;
    }

    if (args->arg_type[0] != STRING_RESULT ||
        args->arg_type[1] != STRING_RESULT ||
        args->arg_type[2] != STRING_RESULT) {
        snprintf(message, MYSQL_ERRMSG_SIZE, "All arguments must be strings");
        return true;
    }

    initid->maybe_null = true;
    initid->max_length = 1024 * 1024; // 1MB max
    initid->ptr = nullptr;

    return false;
}

static void generate_embeddings_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        delete[] initid->ptr;
        initid->ptr = nullptr;
    }
}

static int parse_json_string_array(const char *json, size_t json_len,
                                   char ***out_strings, size_t **out_lengths, size_t *out_count) {
    size_t capacity = 10;
    char **strings = new char*[capacity];
    size_t *lengths = new size_t[capacity];
    size_t count = 0;

    const char *p = json;
    const char *end = json + json_len;

    // Skip whitespace and opening bracket
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (p >= end || *p != '[') {
        delete[] strings;
        delete[] lengths;
        return -1;
    }
    p++;

    while (p < end) {
        // Skip whitespace
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;

        if (p >= end) break;
        if (*p == ']') break;
        if (*p == ',') {
            p++;
            continue;
        }

        // Expect a quoted string
        if (*p != '"') {
            for (size_t i = 0; i < count; i++) delete[] strings[i];
            delete[] strings;
            delete[] lengths;
            return -1;
        }
        p++;

        const char *str_start = p;
        size_t str_len = 0;

        // Find end of string
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                p++; // Skip escaped char
            }
            p++;
            str_len++;
        }

        if (p >= end) {
            for (size_t i = 0; i < count; i++) delete[] strings[i];
            delete[] strings;
            delete[] lengths;
            return -1;
        }

        // Expand array if needed
        if (count >= capacity) {
            capacity *= 2;
            char **new_strings = new char*[capacity];
            size_t *new_lengths = new size_t[capacity];
            memcpy(new_strings, strings, sizeof(char*) * count);
            memcpy(new_lengths, lengths, sizeof(size_t) * count);
            delete[] strings;
            delete[] lengths;
            strings = new_strings;
            lengths = new_lengths;
        }

        // Copy string
        char *str_copy = new char[str_len + 1];
        memcpy(str_copy, str_start, str_len);
        str_copy[str_len] = '\0';

        strings[count] = str_copy;
        lengths[count] = str_len;
        count++;

        p++; // Skip closing quote
    }

    *out_strings = strings;
    *out_lengths = lengths;
    *out_count = count;
    return 0;
}

static char *generate_embeddings(UDF_INIT *initid, UDF_ARGS *args,
                                       char * /*result*/, unsigned long *length,
                                       unsigned char *is_null, unsigned char *error) {
    const char *method = args->args[0];
    const char *model = args->args[1];
    const char *texts_json = args->args[2];

    if (!method || !model || !texts_json) {
        *is_null = 1;
        return nullptr;
    }

    // Validate method
    int method_id = validate_embedding_method(method);
    if (method_id < 0) {
        *error = 1;
        log_message(ERROR_LEVEL, "Invalid embedding method in batch");
        return nullptr;
    }

    // Validate model
    int model_id = validate_embedding_model(method_id, model, INPUT_TYPE_TEXT);
    if (model_id < 0) {
        *error = 1;
        log_message(ERROR_LEVEL, "Invalid or unsupported model in batch");
        return nullptr;
    }

    // Parse JSON array
    char **strings = nullptr;
    size_t *string_lengths = nullptr;
    size_t n_strings = 0;

    if (parse_json_string_array(texts_json, args->lengths[2], &strings, &string_lengths, &n_strings) != 0) {
        *error = 1;
        log_message(ERROR_LEVEL, "Failed to parse JSON array");
        return nullptr;
    }

    if (n_strings == 0) {
        *is_null = 1;
        return nullptr;
    }

    // Prepare inputs
    StringSlice *inputs = new StringSlice[n_strings];
    for (size_t i = 0; i < n_strings; i++) {
        inputs[i].ptr = strings[i];
        inputs[i].len = string_lengths[i];
    }

    // Generate embeddings
    EmbeddingBatch batch;
    int err = generate_embeddings_from_texts(method_id, model_id, inputs, n_strings, &batch);

    // Cleanup input strings
    for (size_t i = 0; i < n_strings; i++) {
        delete[] strings[i];
    }
    delete[] strings;
    delete[] string_lengths;
    delete[] inputs;

    if (err != 0) {
        free_embedding_batch(&batch);
        *error = 1;
        log_message(ERROR_LEVEL, "Batch embedding generation failed");
        return nullptr;
    }

    // Format output as JSON array of arrays
    size_t json_capacity = 1024 * 1024; // 1MB
    char *json_output = new char[json_capacity];
    size_t json_len = 0;

    json_len += snprintf(json_output + json_len, json_capacity - json_len, "[");

    for (size_t i = 0; i < batch.n_vectors; i++) {
        if (i > 0) {
            json_len += snprintf(json_output + json_len, json_capacity - json_len, ",");
        }
        json_len += snprintf(json_output + json_len, json_capacity - json_len, "[");

        for (size_t j = 0; j < batch.dim; j++) {
            if (j > 0) {
                json_len += snprintf(json_output + json_len, json_capacity - json_len, ",");
            }
            json_len += snprintf(json_output + json_len, json_capacity - json_len,
                               "%.6f", batch.data[i * batch.dim + j]);

            if (json_len > json_capacity - 1000) {
                delete[] json_output;
                free_embedding_batch(&batch);
                *error = 1;
                log_message(ERROR_LEVEL, "Output too large for batch");
                return nullptr;
            }
        }

        json_len += snprintf(json_output + json_len, json_capacity - json_len, "]");
    }

    json_len += snprintf(json_output + json_len, json_capacity - json_len, "]");

    free_embedding_batch(&batch);

    // Store result
    if (initid->ptr) {
        delete[] initid->ptr;
    }
    initid->ptr = json_output;
    *length = json_len;

    return json_output;
}

/* Component initialization */
static mysql_service_status_t component_mysql_gembed_init() {
    log_message(INFORMATION_LEVEL, "initializing...");

    // Register GENERATE_EMBEDDING UDF
    if (mysql_service_udf_registration->udf_register(
            "GENERATE_EMBEDDING",
            Item_result::STRING_RESULT,
            (Udf_func_any)generate_embedding,
            generate_embedding_init,
            generate_embedding_deinit)) {
        log_message(ERROR_LEVEL, "Failed to register GENERATE_EMBEDDING");
        return 1;
    }

    // Register GENERATE_EMBEDDINGS UDF
    if (mysql_service_udf_registration->udf_register(
            "GENERATE_EMBEDDINGS",
            Item_result::STRING_RESULT,
            (Udf_func_any)generate_embeddings,
            generate_embeddings_init,
            generate_embeddings_deinit)) {
        log_message(ERROR_LEVEL, "Failed to register GENERATE_EMBEDDINGS");
        mysql_service_udf_registration->udf_unregister("GENERATE_EMBEDDING", nullptr);
        return 1;
    }

    log_message(INFORMATION_LEVEL, "functions registered successfully");
    return 0;
}

/* Component deinitialization */
static mysql_service_status_t component_mysql_gembed_deinit() {
    log_message(INFORMATION_LEVEL, "shutting down...");

    int was_present = 0;
    mysql_service_udf_registration->udf_unregister("GENERATE_EMBEDDING", &was_present);
    mysql_service_udf_registration->udf_unregister("GENERATE_EMBEDDINGS", &was_present);

    log_message(INFORMATION_LEVEL, "functions unregistered");
    return 0;
}

/* Component declaration */
DECLARE_COMPONENT(component_mysql_gembed, "mysql:component_mysql_gembed")
    component_mysql_gembed_init,
    component_mysql_gembed_deinit
END_DECLARE_COMPONENT();

DECLARE_LIBRARY_COMPONENTS
    &COMPONENT_REF(component_mysql_gembed)
END_DECLARE_LIBRARY_COMPONENTS
