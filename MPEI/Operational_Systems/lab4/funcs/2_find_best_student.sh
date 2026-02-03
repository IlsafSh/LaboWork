#!/bin/bash

#trap 'echo "# $BASH_COMMAND";read' DEBUG

# Функция для нахождения студента с максимальным числом правильных ответов
find_best_student() {
  local fs_path=${1}
  local group=${!#}
  declare -A student_ans

  # Поиск всех тестовых файлов
  tests_files=$(find $fs_path/ -type f -path "*/tests/TEST-*")
  #echo $tests_files

  # Подсчет общего количества правильных ответов для каждого студента
  for test_file in $tests_files; do
    while IFS=',' read -r year student grp corr_ans score; do
      if [ "$grp" == "$group" ] && [ "$score" -ge 3 ]; then
        student_ans["$student"]=$((student_ans["$student"] + corr_ans))
      fi
    done < "$test_file"
  done

  # Определение студента с максимальным количеством баллов
  local best_student=""
  local max_ans=0
  for student in "${!student_ans[@]}"; do
    if [ "${student_ans[$student]}" -gt "$max_ans" ]; then
      best_student=$student
      max_ans=${student_ans[$student]}
    fi
  done

  echo "Лучший студент: $best_student с общим баллом $max_ans"
}

find_best_student "$@"
