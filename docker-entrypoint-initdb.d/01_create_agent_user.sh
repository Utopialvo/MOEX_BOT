# Файл: MOEX_BOT/docker-entrypoint-initdb.d/01_create_agent_user.sh
#!/bin/bash
set -e

# Создаём пользователя agent_ro с sha256_password, пароль из переменной окружения
clickhouse-client --user "${CLICKHOUSE_USER}" --password "${CLICKHOUSE_PASSWORD}" --query "
CREATE USER IF NOT EXISTS agent_ro IDENTIFIED WITH sha256_password BY '${AGENT_READONLY_PASSWORD}';
GRANT SELECT ON default.* TO agent_ro;
"