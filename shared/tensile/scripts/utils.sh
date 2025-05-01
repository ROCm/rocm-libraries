ERR='\033[0;31m'
INFO='\033[0;32m'
NC='\033[0m' # No color
echoerr() { echo -e "${ERR}$@${NC}" 1>&2; }
echoinfo() { echo -e "${INFO}$@${NC}" 1>&2; }

convert_comma_separated_to_array() {
    local var="$1"
    if [[ "$var" == *,* ]]; then
        IFS=',' read -r -a array <<< "$var"
        echo "${array[@]}"
    else
        echo "$var"
    fi
}

assert_envvar_exists() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echoerr "Error: Environment variable $var_name is not set, see \`--help\`."
        exit 1
    fi
}