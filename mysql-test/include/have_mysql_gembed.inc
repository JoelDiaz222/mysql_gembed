disable_query_log;

#
# Check if the mysql_gembed component is available.
#
if (!$MYSQL_GEMBED) {
   --skip mysql_gembed requires the environment variable \$MYSQL_GEMBED to be set (normally done by mtr), see the file plugin.defs
}

#
# Check if --plugin-dir was setup for mysql_gembed.
#
if (`SELECT CONCAT('--plugin-dir=', REPLACE(@@plugin_dir, '\\\\', '/')) != '$MYSQL_GEMBED_OPT/'`) {
    --skip mysql_gembed requires that --plugin-dir is set to the component dir (the .opt file does not contain \$MYSQL_GEMBED_OPT)
}

enable_query_log;
