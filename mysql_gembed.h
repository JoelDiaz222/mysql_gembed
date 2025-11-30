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
#ifndef MYSQL_GEMBED_H
#define MYSQL_GEMBED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INPUT_TYPE_TEXT 0
#define INPUT_TYPE_IMAGE 1
#define INPUT_TYPE_MULTIMODAL 2

/* Structure for storing generated embeddings */
typedef struct
{
    float *data;
    size_t n_vectors;
    size_t dim;
} EmbeddingBatch;

/* Structure for passing text data */
typedef struct
{
    const char *ptr;
    size_t len;
} StringSlice;

/* Structure for passing binary data */
typedef struct
{
    const unsigned char *ptr;
    size_t len;
} ByteSlice;

/* Generic input data structure */
typedef struct
{
    int input_type;                    /* INPUT_TYPE_* constant */
    const ByteSlice *binary_data;      /* For images, audio, etc. */
    size_t n_binary;                   /* Number of binary items */
    const StringSlice *text_data;      /* For text inputs */
    size_t n_text;                     /* Number of text items */
} InputData;

/* Validates the embedding method name and returns method ID */
extern int validate_embedding_method(const char *method);

/* Validates the model name for a given method and returns model ID */
extern int validate_embedding_model(int method_id, const char *model, int input_type);

/* Generates embeddings for the given input data */
extern int generate_embeddings(
    int method_id,
    int model_id,
    const InputData *input_data,
    EmbeddingBatch *out_batch
);

/* Frees memory allocated for an embedding batch */
extern void free_embedding_batch(EmbeddingBatch *batch);

#ifdef __cplusplus
}
#endif

#endif /* MYSQL_GEMBED_H */
