#!/bin/bash

echo "this will work"
RESULT=$?
if [[ ${RESULT} -eq 0 ]]; then
  echo success
else
  echo failed
fi

if [[ ${RESULT} == 0 ]]; then
  echo success 2
else
  echo failed 2
fi
