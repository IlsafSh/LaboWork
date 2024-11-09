#!/bin/bash

#trap 'echo "# $BASH_COMMAND";read' DEBUG

# Функция проверки посещаемости и оценок
check_attendance_and_scores() {
  local fs_path=${1}
  local group=${3}
  local subject=${4}
  local subject_folder="$fs_path/$subject"
  local tests_folder="$subject_folder/tests"
  local attendance_file="$subject_folder/$group-attendance"
  local result="Студенты, пропустившие 1-ую лекцию и сдавшие 1-ый тест на 5:\n"

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

  # Проверка, что файл посещаемости предмета для группы существует
  if [ ! -f "$attendance_file" ]; then
    echo "Файл с посещаемостью группы $attendance_file не найден"
    return
  fi

  # Проверка студентов по посещаемости и оценке за первый тест
  while IFS=' ' read -r student attendance; do
    # Проверка посещаемости первой лекции
    if [[ "${attendance:0:1}" == "_" ]]; then
      for test_file in "$tests_folder"/TEST-1; do
        # Чтение строк файла теста
        while IFS=',' read -r year stud grp corr_ans score; do
          # Если это нужный студент из нужной группы
          if [[ "$grp" == "$group" && "$stud" == "$student" ]]; then
            if (( score == 5 )); then
              result+="$student пропустил первую лекцию и сдал первый тест на 5\n"
              break  # Тест считается сданным на отлично, выходим из цикла
            fi
          fi
        done < "$test_file"
      done
    fi
  done < "$attendance_file"

  # Если в result ничего не добавлено, значит условие не выполнено
  if [[ "$result" == "Студенты, пропустившие 1-ую лекцию и сдавшие 1-ый тест на 5:\n" ]]; then
    result+="Студенты не найдены."
  fi

  echo -e "$result"
}

check_attendance_and_scores "$@"
