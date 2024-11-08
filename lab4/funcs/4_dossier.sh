#!/bin/bash

#trap 'echo "# $BASH_COMMAND";read' DEBUG

# Функция для вывода досье студента
display_student_dossier() {
  local student=${!#}
  local notes_folder="./students/general/notes"
  local result="Досье на студента $student:\n"

   # Проверка, что папка с записями существует
  if [ ! -d "$notes_folder" ]; then
    echo "Папка с записями не найдена"
    return
  fi

  # Поиск по всем файлам *Names.log
  for notes_file in "$notes_folder"/*.log; do
    if grep -q "$student" "$notes_file"; then
      result+="$(grep -A 1 "$student" "$notes_file")\n"
    fi
  done

   # Если в result ничего не добавлено, значит студент не найден
  if [[ "$result" == "Досье на студента $student:\n" ]]; then
    result+="Студент не найден."
  fi

  echo -e "$result"
}

display_student_dossier "$@"