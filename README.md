# mysql_gembed

## Generate Embeddings directly in MySQL

## 1. Prepare the Environment

```bash
git clone git@github.com:mysql/mysql-server.git
cd mysql-server/components
git clone git@github.com:JoelDiaz222/mysql_gembed.git
```

## 2. Build

Assumes you are in the `mysql-server` directory. This configuration points to Homebrew's Bison, but you can change it to a different executable.

```bash
mkdir build
cd build

cmake .. -DBISON_EXECUTABLE=/opt/homebrew/opt/bison/bin/bison \
  -DWITH_UNIT_TESTS=OFF \
  -DWITH_EDITLINE=bundled

make -j$(sysctl -n hw.ncpu)
```

## 3. Install & Initialize

```bash
sudo make install

sudo /usr/local/mysql/bin/mysqld --initialize \
  --basedir=/usr/local/mysql \
  --datadir=/usr/local/mysql/data
```

## 4. Run Server

Start `mysqld_safe` in the background.

```bash
/usr/local/mysql/bin/mysqld_safe \
  --datadir=/usr/local/mysql/data \
  --socket=/tmp/mysql.sock &
```

## 5. Usage

### Connect
```bash
/usr/local/mysql/bin/mysql -u root
```

### Vector Embeddings (SQL)

**Generate Single Embedding:**

```sql
SELECT VECTOR_TO_STRING(
    EMBED_TEXT("fastembed", "Qdrant/all-MiniLM-L6-v2-onnx", "a")
) AS readable_embedding;
```

**Generate Batch Embeddings:**

```sql
SELECT EMBED_TEXTS(
    'fastembed',
    'Qdrant/all-MiniLM-L6-v2-onnx',
    '["hello", "world", "test"]'
) AS embeddings;
```

**Pretty Print Embeddings:**

```sql
SELECT JSON_PRETTY(
    CONVERT(
        EMBED_TEXTS(
            'fastembed',
            'Qdrant/all-MiniLM-L6-v2-onnx',
            '["hello", "world", "test"]'
        ) USING utf8mb4)
) AS readable_embeddings;
```

## 6. Stop Server

```bash
sudo pkill mysqld
```
