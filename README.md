# mysql_gembed

A MySQL server component that brings **in-database embedding generation** directly into MySQL, by using the
[Gembed Rust core](https://github.com/JoelDiaz222/gembed).

The component is a thin adapter that marshals MySQL types into the C ABI of the Gembed Rust core (`libgembed`), which
handles model loading and inference.

## Architecture

```
┌───────────────────────────────────────────────┐
│                MySQL SQL Query                │
│         (e.g. SELECT EMBED_TEXT(...))         │
└───────────────────────┬───────────────────────┘
                        │ MySQL Component API
                        ▼
┌───────────────────────────────────────────────┐
│        MySQL Component (mysql_gembed)         │
│  - Registers EMBED_TEXT / EMBED_TEXTS UDFs    │
│  - Marshals MySQL types → C ABI types         │
└───────────────────────┬───────────────────────┘
                        │ C FFI
                        ▼
┌───────────────────────────────────────────────┐
│         Rust Core Library (libgembed)         │
│          Contains embedding backends          │
└───────────────────────────────────────────────┘
```

## Build

Clone this repository **inside** the MySQL source tree, then build MySQL normally.

```bash
git clone git@github.com:mysql/mysql-server.git
cd mysql-server/components
git clone git@github.com:JoelDiaz222/mysql_gembed.git --recurse-submodules
```

From the `mysql-server` directory:

```bash
mkdir build && cd build

cmake .. \
  -DBISON_EXECUTABLE=/opt/homebrew/opt/bison/bin/bison \
  -DWITH_UNIT_TESTS=OFF \
  -DWITH_EDITLINE=bundled

make -j$(nproc)
```

## Install & Run

```bash
sudo make install

# Initialize and start the server
sudo /usr/local/mysql/bin/mysqld --initialize \
  --basedir=/usr/local/mysql \
  --datadir=/usr/local/mysql/data

/usr/local/mysql/bin/mysqld_safe \
  --datadir=/usr/local/mysql/data \
  --socket=/tmp/mysql.sock &
```

## Usage

```bash
/usr/local/mysql/bin/mysql -u root
```

```sql
-- Load the component
INSTALL COMPONENT 'file://component_mysql_gembed';

-- Embed a single string
SELECT VECTOR_TO_STRING(
    EMBED_TEXT('embed_anything', 'Qdrant/all-MiniLM-L6-v2-onnx', 'Hello world')
) AS embedding;

-- Embed a batch of strings (JSON array)
SELECT JSON_PRETTY(CONVERT(
    EMBED_TEXTS(
        'embed_anything',
        'Qdrant/all-MiniLM-L6-v2-onnx',
        '["hello", "world", "test"]'
    ) USING utf8mb4
)) AS embeddings;
```

## Stop Server

```bash
sudo pkill mysqld
```
