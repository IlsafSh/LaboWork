#!/bin/bash

# Проверка, установлен ли dialog
if ! command -v dialog &> /dev/null; then
    echo "dialog не установлен. Пожалуйста, установите его и повторите попытку."
    exit 1
fi

DIALOG=dialog
tempfile=$(mktemp)  # Временный файл для вывода результатов

# Функция для выбора года группы в зависимости от ее номера
select_group() {
  local category=$1
  local group_choice

  case $category in
    "A-06-xx")
      group_choice=$($DIALOG --clear --stdout --title "Выбор группы A-06-xx" \
        --radiolist "Выберите группу:" 20 50 16 \
        "A-06-04" "" off \
        "A-06-05" "" off \
        "A-06-06" "" off \
        "A-06-07" "" off \
        "A-06-08" "" off \
        "A-06-09" "" off \
        "A-06-10" "" off \
        "A-06-11" "" off \
        "A-06-12" "" off \
        "A-06-13" "" off \
        "A-06-14" "" off \
        "A-06-15" "" off \
        "A-06-16" "" off \
        "A-06-17" "" off \
        "A-06-18" "" off \
        "A-06-19" "" off \
        "A-06-20" "" off \
        "A-06-21" "" off)
      ;;
    "A-09-xx")
      group_choice=$($DIALOG --clear --stdout --title "Выбор группы A-09-xx" \
        --radiolist "Выберите группу:" 20 50 5 \
        "A-09-17" "" off \
        "A-09-18" "" off \
        "A-09-19" "" off \
        "A-09-20" "" off \
        "A-09-21" "" off)
      ;;
    "Ae-21-xx")
      group_choice=$($DIALOG --clear --stdout --title "Выбор группы Ae-21-xx" \
        --radiolist "Выберите группу:" 10 30 1 \
        "Ae-21-21" "" off)
      ;;
    *)
      $DIALOG --msgbox "Неизвестная категория." 10 30
      return 1
      ;;
  esac

  # Проверка, была ли выбрана группа
  if [ -z "$group_choice" ]; then
    return 1
  fi

  echo "$group_choice"
}

# Функция для выбора категории группы
select_group_category() {
  local group_category_choice=$($DIALOG --clear --stdout --title "Выбор категории групп" \
    --menu "Выберите категорию:" 15 50 3 \
    1 "A-06-xx" \
    2 "A-09-xx" \
    3 "Ae-21-xx")

  case $group_category_choice in
    1) group_category="A-06-xx" ;;
    2) group_category="A-09-xx" ;;
    3) group_category="Ae-21-xx" ;;
    *) return 1 ;;
  esac

  # Получение выбора группы
  group_choice=$(select_group "$group_category")

  # Проверка успешности выбора группы
  if [[ $? -ne 0 ]]; then
    return 1
  fi

  echo "$group_choice"
}

# Функция для выбора предмета
select_subject() {
  local subject_choice=$($DIALOG --clear --stdout --title "Выбор предмета" \
    --radiolist "Выберите предмет:" 15 50 2 \
    "Пивоварение" "" off \
    "Уфология" "" off)

  # Проверка, был ли выбран предмет
  if [ -z "$subject_choice" ]; then
    return 1
  fi

  echo "$subject_choice"
}

# Функция проверки выбора категории с выводом предупреждающего окна
check_category() {
  local selection=$1
  local item_name=$2

  if [ -z "$selection" ]; then
    $DIALOG --msgbox "Вы не выбрали $item_name!" 10 40
    return 1
  fi

  return 0
}

# Основной цикл для возврата в главное меню
while true; do
  choice=$($DIALOG --clear --stdout --title "Главное меню" \
    --ok-label "Выбрать" --cancel-label "Выйти" \
    --menu "Выберите функцию:" 15 70 5 \
    1 "Поиск студентов, не сдавших тесты" \
    2 "Лучший студент по общему числу правильных ответов" \
    3 "Средний балл по предмету" \
    4 "Досье студента" \
    5 "Проверка посещаемости и оценок")

  if [ $? -ne 0 ]; then
    clear
    echo "Работа завершена."
    rm "$tempfile"  # Удаляем временный файл перед завершением
    exit 0
  fi

  case $choice in
    1) # Поиск студентов, не сдавших тесты
      group_choice=$(select_group_category)
      check_category "$group_choice" "группу" || continue

      subject_choice=$(select_subject)
      check_category "$subject_choice" "предмет" || continue

      bash funcs/1_search_students.sh search_students_not_passed_tests "$group_choice" "$subject_choice" > "$tempfile"
      fold -s -w 160 "$tempfile" > "${tempfile}_folded"
      $DIALOG --title "Студенты, не сдавшие тесты" --textbox "${tempfile}_folded" 20 100
      rm "${tempfile}_folded"  # Удаляем временный отформатированный файл
    ;;

    2) # Лучший студент по числу правильных ответов
      group_choice=$(select_group_category)
      check_category "$group_choice" "группу" || continue

      bash funcs/2_find_best_student.sh find_best_student "$group_choice" > "$tempfile"
      fold -s -w 160 "$tempfile" > "${tempfile}_folded"
      $DIALOG --title "Лучший студент" --textbox "${tempfile}_folded" 20 100
      rm "${tempfile}_folded"
    ;;

    3) # Средний балл по предмету
      subject_choice=$(select_subject)
      check_category "$subject_choice" "предмет" || continue

      student=$($DIALOG --inputbox "Введите фамилию студента:" 10 40 --stdout)
      check_category "$student" "студента" || continue

      bash funcs/3_average_score.sh calculate_average_score "$subject_choice" "$student" > "$tempfile"
      fold -s -w 160 "$tempfile" > "${tempfile}_folded"
      $DIALOG --title "Средний балл студента" --textbox "${tempfile}_folded" 20 100
      rm "${tempfile}_folded"
    ;;

    4) # Досье студента
      student=$($DIALOG --inputbox "Введите фамилию студента:" 10 40 --stdout)
      check_category "$student" "студента" || continue

      bash funcs/4_dossier.sh display_student_dossier "$student" > "$tempfile"
      fold -s -w 160 "$tempfile" > "${tempfile}_folded"
      $DIALOG --title "Досье студента" --textbox "${tempfile}_folded" 20 100
      rm "${tempfile}_folded"
    ;;

    5) # Проверка посещаемости и оценок
      group_choice=$(select_group_category)
      check_category "$group_choice" "группу" || continue

      subject_choice=$(select_subject)
      check_category "$subject_choice" "предмет" || continue

      bash funcs/5_attendance_and_scores.sh check_attendance_and_scores "$group_choice" "$subject_choice" > "$tempfile"
      fold -s -w 160 "$tempfile" > "${tempfile}_folded"
      $DIALOG --title "Посещаемость и оценки" --textbox "${tempfile}_folded" 20 100
      rm "${tempfile}_folded"
    ;;
  esac
done

# Удаляем временный файл при выходе из цикла
rm "$tempfile"

$DIALOG --clear
echo "Работа завершена."
