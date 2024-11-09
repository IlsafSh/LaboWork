#!/bin/bash

#trap 'echo "# $BASH_COMMAND";read' DEBUG

# Функция для вычисления среднего балла по предмету для студента с частичным совпадением имени
calculate_average_score() {
  local fs_path=${1}
  local subject=${3}
  local partial_name=${4}
  local subject_folder="$fs_path/$subject"
  local tests_folder="$subject_folder/tests"
  local total_score=0
  local count=0

  # Проверка, что папка предмета существует
  if [ ! -d "$subject_folder" ]; then
    echo "Папка предмета $subject не найдена"
    return
  fi

  # Проверка, что папка тестов предмета существует
  if [ ! -d "$tests_folder" ]; then
    echo "Папка с тестами предмета $subject не найдена"
    return
  fi

  # Подсчет среднего балла для указанного студента
  for test_file in "$tests_folder"/TEST-*; do
    while IFS=',' read -r year stud grp corr_ans score; do
      # Проверяем, совпадает ли имя студента с частичным совпадением
      if [[ "$stud" == *"$partial_name"* ]]; then
        total_score=$((total_score + score))
        count=$((count + 1))
      fi
    done < "$test_file"
  done

  if [ "$count" -gt 0 ]; then
    local average=$(echo "scale=2; $total_score / $count" | bc)
    echo "Средний балл для студента, имя которого содержит '$partial_name', по предмету $subject: $average"
  else
    echo "Нет данных для студента с именем, содержащим '$partial_name', по предмету $subject"
  fi
}

calculate_average_score "$@"
