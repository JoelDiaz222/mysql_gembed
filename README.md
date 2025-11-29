# mysql_gembed

## Generate Embeddings directly in MySQL

## 1. Build
Assumes you are in the `mysql-server` source code, and the component's files are in a directory inside `mysql-server/components/`. This configuration points to Homebrew's Bison, but you can change it to a different executable.

```bash
mkdir build
cd build

# Configure
cmake .. -DBISON_EXECUTABLE=/opt/homebrew/opt/bison/bin/bison \
  -DWITH_UNIT_TESTS=OFF \
  -DWITH_EDITLINE=bundled

# Compile
make -j$(sysctl -n hw.ncpu)
```

## 2. Install & Initialize

```bash
sudo make install

sudo /usr/local/mysql/bin/mysqld --initialize \
  --basedir=/usr/local/mysql \
  --datadir=/usr/local/mysql/data
```

## 3. Run Server
Start `mysqld_safe` in the background.

```bash
/usr/local/mysql/bin/mysqld_safe \
  --datadir=/usr/local/mysql/data \
  --socket=/tmp/mysql.sock &
```

## 4. Usage

### Connect
```bash
/usr/local/mysql/bin/mysql -u root
```

### Vector Embeddings (SQL)

**Generate Single Embedding:**
```sql
SELECT VECTOR_TO_STRING(
    GENERATE_EMBEDDING("fastembed", "Qdrant/all-MiniLM-L6-v2-onnx", "a")
) AS readable_embedding;
```

**Generate Batch Embeddings:**
```sql
SELECT GENERATE_EMBEDDINGS(
    'fastembed',
    'Qdrant/all-MiniLM-L6-v2-onnx',
    '["hello", "world", "test"]'
) AS embeddings;
```

**Pretty Print Embeddings:**
```sql
SELECT JSON_PRETTY(
    CONVERT(
        GENERATE_EMBEDDINGS(
            'fastembed',
            'Qdrant/all-MiniLM-L6-v2-onnx',
            '["hello", "world", "test"]'
        ) USING utf8mb4)
) AS readable_embeddings;
```

## 5. Stop Server
```bash
sudo pkill mysqld
```
