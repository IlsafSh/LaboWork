org  0x7C00              ; загрузка загрузочного сектора по адресу 0х7с00
start:
     mov  si, message
     mov  ah, 0x0e       ; записываем в ah значение 0х0е для печати на экран

loop1:
     lodsb               ; загружаем очередной символ в al
     cmp al, 0           ; проверка (нулевой символ означает конец строки)
     jz   loop_exit      ; если все закончилось, он прыгает на бесконечный цикл
     int  0x10           ; иначе вызываем прерывание для работы с экранмом
     jmp  loop1          ; и прыгаем опять обратно на загрузку символов

loop_exit:
     jmp  $              ; постоянно прыгает на адрес текущей строки в памяти (бесконечный цикл)
 
message:
     db  " ____  _____   ____   _______    _________       ____    _______ ", 10, 13, "|_   \|_   _|.'    \.|_   __ \  |  _   _  |    .'    \. /  ___  |", 10, 13, "  |   \ | | /  .--.  \ | |__) | |_/ | | \_|   /  .--.  \  (__ \_|", 10, 13, "  | |\ \| | | |    | | |  __ /      | |       | |    | |'.___\-. ", 10, 13, " _| |_\   |_\  \--'  /_| |  \ \_   _| |_      \  \--'  /\\____) |", 10, 13, "|_____|\____|\.____.'|____| |___| |_____|      \.____.'|_______.'", 0
                                                           
; ____  _____   ____   _______    _________       ____    _______ 
;|_   \|_   _|.'    \.|_   __ \  |  _   _  |    .'    \. /  ___  |
;  |   \ | | /  .--.  \ | |__) | |_/ | | \_|   /  .--.  \  (__ \_|
;  | |\ \| | | |    | | |  __ /      | |       | |    | |'.___\-. 
; _| |_\   |_\  \--'  /_| |  \ \_   _| |_      \  \--'  /\\____) |
;|_____|\____|\.____.'|____| |___| |_____|      \.____.'|_______.'
                                                                                                                                       
ex:
     times 510 - ($ - $$) db 0     ; добавление в машинный код нулей до адреса 510
     dw 0xaa55                     ; запись magic number  (wow o_o)