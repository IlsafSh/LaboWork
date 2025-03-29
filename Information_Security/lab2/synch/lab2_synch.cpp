#include <botan/block_cipher.h>
#include <botan/hex.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <windows.h>

// Функция для конвертации std::string → std::wstring
std::wstring to_wstring(const std::string& str) {
    return std::wstring(str.begin(), str.end());
}

void pad_buffer(std::vector<uint8_t>& buffer, size_t block_size) {
    size_t padding_size = block_size - (buffer.size() % block_size);
    buffer.insert(buffer.end(), padding_size, static_cast<uint8_t>(padding_size));
}

void unpad_buffer(std::vector<uint8_t>& buffer) {
    if (!buffer.empty()) {
        uint8_t padding_size = buffer.back();
        if (padding_size <= buffer.size()) {
            buffer.resize(buffer.size() - padding_size);
        }
    }
}

void encrypt_file(const std::string& input_filename, const std::string& output_filename, const std::vector<uint8_t>& key) {
    std::unique_ptr<Botan::BlockCipher> cipher(Botan::BlockCipher::create("GOST-28147-89"));
    if (!cipher) {
        throw std::runtime_error("Ошибка: не удалось создать объект шифра GOST-28147-89");
    }
    cipher->set_key(key);

    std::ifstream input(input_filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Ошибка: не удалось открыть входной файл");
    }
    std::ofstream output(output_filename, std::ios::binary);

    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(input)), {});
    pad_buffer(buffer, cipher->block_size());

    for (size_t i = 0; i < buffer.size(); i += cipher->block_size()) {
        cipher->encrypt(&buffer[i]);
    }
    output.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

void decrypt_file(const std::string& input_filename, const std::string& output_filename, const std::vector<uint8_t>& key) {
    std::unique_ptr<Botan::BlockCipher> cipher(Botan::BlockCipher::create("GOST-28147-89"));
    if (!cipher) {
        throw std::runtime_error("Ошибка: не удалось создать объект шифра GOST-28147-89");
    }
    cipher->set_key(key);

    std::ifstream input(input_filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Ошибка: не удалось открыть зашифрованный файл");
    }
    std::ofstream output(output_filename, std::ios::binary);

    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(input)), {});

    for (size_t i = 0; i < buffer.size(); i += cipher->block_size()) {
        cipher->decrypt(&buffer[i]);
    }
    unpad_buffer(buffer);
    output.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

std::string get_filename_from_user(const std::wstring& prompt, const std::string& default_name) {
    std::wcout << prompt;
    std::string filename;
    std::getline(std::cin, filename);
    return filename.empty() ? default_name : filename;
}

int main() {
#ifdef BOTAN_HAS_GOST_28147_89
    try {
        setlocale(LC_ALL, "ru_RU.UTF-8");
        SetConsoleOutputCP(CP_UTF8);

        std::vector<uint8_t> key = Botan::hex_decode(
            "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF");
        std::wcout << L"----------Симметричное шифрование----------\n";
        std::wcout << L"-Botan---------GOST-28147-89---------Magma-\n";
        while (true) {
            std::wcout << L"\nВыберите действие:\n"
                << L"1 - Зашифровать файл\n"
                << L"2 - Расшифровать файл\n"
                << L"0 - Выход\n"
                << L"...";

            int choice;
            std::cin >> choice;
            std::cin.ignore(); // Очистка буфера после ввода числа

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
