#!/bin/bash

IN_FILE=$1

sed -i -E "s/u\. s( |$)/u. s./g" $IN_FILE
sed -i -E "s/([]]) s( |$)/\1 's\2/g" $IN_FILE
sed -i -E "s/([^]]) s( |$)/\1's\2/g" $IN_FILE

