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

## Test

The component includes one MTR integration test case,
`mysql_gembed.mysql_gembed`. It starts a real MySQL server, loads the component,
calls `EMBED_TEXT` and `EMBED_TEXTS`, validates vector dimensions and JSON batch
shape, checks null/error paths, and unloads the component.

From the `mysql-server` directory, first build the component:

```bash
cmake --build build --target component_mysql_gembed -j$(nproc)
```

Copy the bundled MTR files into the MySQL test tree:

```bash
cp components/mysql_gembed/mysql-test/include/have_mysql_gembed.inc \
  mysql-test/include/

cp -R components/mysql_gembed/mysql-test/suite/mysql_gembed \
  mysql-test/suite/
```

Register the component with MTR by adding this line to
`mysql-test/include/plugin.defs`:

```text
component_mysql_gembed          plugin_output_directory  no  MYSQL_GEMBED
```

The same line is available in
[mysql_gembed.defs](mysql-test/include/plugin.defs.d/mysql_gembed.defs).

Finally, run the component suite from the build tree:

```bash
cd build/mysql-test
./mtr --suite=mysql_gembed mysql_gembed
```

Expected result:

```text
mysql_gembed.mysql_gembed                 [ pass ]
shutdown_report                           [ pass ]
Completed: All 2 tests were successful.
```

`shutdown_report` is MTR's own cleanup check. The component coverage is the
single `mysql_gembed.mysql_gembed` integration test case.

Inside this repository, the suite files are:

- [mysql_gembed.test](mysql-test/suite/mysql_gembed/t/mysql_gembed.test)
- [mysql_gembed.result](mysql-test/suite/mysql_gembed/r/mysql_gembed.result)
- [mysql_gembed-master.opt](mysql-test/suite/mysql_gembed/t/mysql_gembed-master.opt)

MTR finds the component through `mysql-test/include/plugin.defs`; the suite uses
[mysql_gembed-master.opt](mysql-test/suite/mysql_gembed/t/mysql_gembed-master.opt)
to point `@@plugin_dir` at the build output directory containing
`component_mysql_gembed.so`.

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
