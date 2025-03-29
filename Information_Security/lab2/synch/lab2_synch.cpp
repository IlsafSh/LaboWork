#include <botan/block_cipher.h>  // Библиотека Botan для работы с блочными шифрами
#include <botan/hex.h>           // Библиотека Botan для работы с шестнадцатеричными строками (hex)
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <windows.h>              // Библиотека Windows API для работы с консолью (установка UTF-8)
#include <chrono>                 // Библиотека для замера времени выполнения операций

// Функция конвертации std::string → std::wstring (для корректного вывода в консоль)
std::wstring to_wstring(const std::string& str) {
    return std::wstring(str.begin(), str.end());
}

// Функция добавления паддинга (дополнительных байтов) в конец блока
// Используется для выравнивания данных по размеру блока шифра
void pad_buffer(std::vector<uint8_t>& buffer, size_t block_size) {
    size_t padding_size = block_size - (buffer.size() % block_size);  // Определяем, сколько байтов не хватает до полного блока
    buffer.insert(buffer.end(), padding_size, static_cast<uint8_t>(padding_size)); // Добавляем паддинг, заполняя байтами с числом, равным их количеству
}

// Функция удаления паддинга (используется после расшифрования)
void unpad_buffer(std::vector<uint8_t>& buffer) {
    if (!buffer.empty()) {
        uint8_t padding_size = buffer.back(); // Последний байт указывает, сколько байтов было добавлено при паддинге
        if (padding_size <= buffer.size()) {
            buffer.resize(buffer.size() - padding_size);  // Обрезаем лишние байты
        }
    }
}

// Функция шифрования файла с использованием алгоритма GOST 28147-89 (Магма)
void encrypt_file(const std::string& input_filename, const std::string& output_filename, const std::vector<uint8_t>& key) {
    // Создаем объект блочного шифра GOST 28147-89 (Магма)
    std::unique_ptr<Botan::BlockCipher> cipher(Botan::BlockCipher::create("GOST-28147-89"));
    if (!cipher) {
        throw std::runtime_error("Ошибка: не удалось создать объект шифра GOST-28147-89");
    }
    cipher->set_key(key);  // Устанавливаем ключ шифрования

    // Открываем входной файл для чтения в бинарном режиме
    std::ifstream input(input_filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Ошибка: не удалось открыть входной файл");
    }

    // Открываем выходной файл для записи зашифрованных данных
    std::ofstream output(output_filename, std::ios::binary);

    // Читаем содержимое файла в вектор
    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(input)), {});

    // Добавляем паддинг (чтобы длина данных была кратной размеру блока)
    pad_buffer(buffer, cipher->block_size());

    // Засекаем время начала шифрования
    auto start = std::chrono::high_resolution_clock::now();

    // Шифруем данные поблочно
    for (size_t i = 0; i < buffer.size(); i += cipher->block_size()) {
        cipher->encrypt(&buffer[i]);
    }

    // Засекаем время окончания шифрования
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> encryption_time = end - start;

    // Записываем зашифрованные данные в выходной файл
    output.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    std::wcout << L"Шифрование завершено за " << encryption_time.count() << L" мс.\n";
}

// Функция дешифрования файла
void decrypt_file(const std::string& input_filename, const std::string& output_filename, const std::vector<uint8_t>& key) {
    // Создаем объект блочного шифра GOST 28147-89 (Магма)
    std::unique_ptr<Botan::BlockCipher> cipher(Botan::BlockCipher::create("GOST-28147-89"));
    if (!cipher) {
        throw std::runtime_error("Ошибка: не удалось создать объект шифра GOST-28147-89");
    }
    cipher->set_key(key); // Устанавливаем ключ шифрования

    // Открываем зашифрованный файл для чтения
    std::ifstream input(input_filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Ошибка: не удалось открыть зашифрованный файл");
    }

    // Открываем выходной файл для записи расшифрованных данных
    std::ofstream output(output_filename, std::ios::binary);

    // Читаем содержимое файла в вектор
    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(input)), {});

    // Засекаем время начала дешифрования
    auto start = std::chrono::high_resolution_clock::now();

    // Дешифруем данные поблочно
    for (size_t i = 0; i < buffer.size(); i += cipher->block_size()) {
        cipher->decrypt(&buffer[i]);
    }

    // Засекаем время окончания дешифрования
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decryption_time = end - start;

    // Убираем паддинг (восстанавливаем исходные данные)
    unpad_buffer(buffer);

    // Записываем расшифрованные данные в выходной файл
    output.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    std::wcout << L"Дешифрование завершено за " << decryption_time.count() << L" мс.\n";
}

// Функция запроса имени файла у пользователя
std::string get_filename_from_user(const std::wstring& prompt, const std::string& default_name) {
    std::wcout << prompt;
    std::string filename;
    std::getline(std::cin, filename);
    return filename.empty() ? default_name : filename;
}

int main() {
#ifdef BOTAN_HAS_GOST_28147_89
    try {
        setlocale(LC_ALL, "ru_RU.UTF-8");  // Устанавливаем русскую локаль
        SetConsoleOutputCP(CP_UTF8);       // Устанавливаем кодировку UTF-8 для корректного вывода

        // Определяем ключ шифрования (256 бит)
        std::vector<uint8_t> key = Botan::hex_decode(
            "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF");

        std::wcout << L"----------Симметричное шифрование----------\n";
        std::wcout << L"-Botan---------GOST-28147-89---------Magma-\n";

        int choice;
        while (true) {
            std::wcout << L"\nВыберите действие:\n"
                << L"1 - Зашифровать файл\n"
                << L"2 - Расшифровать файл\n"
                << L"0 - Выход\n"
                << L"...";

            std::cin >> choice;
            std::cin.ignore(); // Очистка буфера после ввода

            if (choice == 1) {
                std::string input_file = get_filename_from_user(
                    L"Введите имя входного файла (по умолчанию: in.txt): ", "in.txt");
                std::string encrypted_file = get_filename_from_user(
                    L"Введите имя файла для сохранения зашифрованных данных (по умолчанию: out_enc.txt): ", "out_enc.txt");

                encrypt_file(input_file, encrypted_file, key);
                std::wcout << L"Файл " << to_wstring(input_file) << L" зашифрован в " << to_wstring(encrypted_file) << std::endl;

            }
            else if (choice == 2) {
                std::string encrypted_file = get_filename_from_user(
                    L"Введите имя зашифрованного файла (по умолчанию: out_enc.txt): ", "out_enc.txt");
                std::string decrypted_file = get_filename_from_user(
                    L"Введите имя файла для сохранения расшифрованных данных (по умолчанию: out_dec.txt): ", "out_dec.txt");

                decrypt_file(encrypted_file, decrypted_file, key);
                std::wcout << L"Файл " << to_wstring(encrypted_file) << L" расшифрован в " << to_wstring(decrypted_file) << std::endl;

                std::wifstream result(decrypted_file);
                result.imbue(std::locale("ru_RU.UTF-8"));
                std::wcout << L"Содержимое расшифрованного файла:\n" << result.rdbuf() << std::endl;

            }
            else {
                std::wcout << L"Работа программы завершена" << std::endl;
                break;
            }
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
#else
    std::cerr << "Ошибка: Поддержка GOST-28147-89 не включена в данной сборке Botan!" << std::endl;
#endif
    return 0;
}
