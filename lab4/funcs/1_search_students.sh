#!/bin/bash

#trap 'echo "# $BASH_COMMAND";read' DEBUG

# 2.1.	Вывод списка студентов, не сдавших хотя бы один тест (с указанием номера теста)
# Функция для поиска студентов, не сдавших тесты
search_students_not_passed_tests() {
  local group=${2}
  local subject=${3}
  local group_file="./students/groups/$group"
  local subject_folder="./$subject"
  local tests_folder="$subject_folder/tests"
  local result="Студенты, не сдавшие хотя бы один тест:\n"

  # Проверка, что файл группы существует
  if [ ! -f "$group_file" ]; then
    echo "Файл со списком студентов группы $group не найден"
    return
  fi

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

  # Считывание списка студентов
  mapfile -t students < "$group_file"

  # Проход по каждому студенту в группе
  for student in "${students[@]}"; do
    # Проход по каждому файлу теста
    for test_file in "$tests_folder"/TEST-*; do
      test_name=$(basename "$test_file")  # Имя теста (например, TEST-1)
      passed=false  # Флаг для отслеживания успешной попытки

      # Чтение строк файла теста
      while IFS=',' read -r year stud grp correct_answers score; do
        # Если это нужный студент из нужной группы
        if [[ "$grp" == "$group" && "$stud" == "$student" ]]; then
          # Если оценка удовлетворительная (>=3), отмечаем тест как сданный
          if (( score >= 3 )); then
            passed=true
            break  # Тест считается сданным, выходим из цикла
          fi
        fi
      done < "$test_file"

      # Если флаг остался ложным, значит студент не сдал этот тест
      if [ "$passed" = false ]; then
        result+="$student не сдал $test_name (последняя оценка < 3)\n"
      fi
    done
  done

  # Если в result ничего не добавлено, значит все студенты сдали тесты
  if [[ "$result" == "Студенты, не сдавшие хотя бы один тест:\n" ]]; then
    result+="Все студенты сдали тесты."
  fi

  echo -e "$result"
}

# Вызов функции с аргументами, переданными в скрипт
search_students_not_passed_tests "$@"
