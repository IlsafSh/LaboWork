#include <botan/auto_rng.h>
#include <botan/hash.h>
#include <botan/hex.h>
#include <botan/gost_3410.h>
#include <botan/data_src.h>
#include <botan/pkcs8.h>
#include <botan/pubkey.h>
#include <botan/x509_key.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <windows.h>

namespace fs = std::filesystem;

void generate_keys(const std::string& key_folder) {
    Botan::AutoSeeded_RNG rng;
    auto group = Botan::EC_Group::from_name("gost_256A");
    auto private_key = Botan::GOST_3410_PrivateKey(rng, group);

    std::ofstream priv_out(key_folder + "/private.key", std::ios::binary);
    priv_out << Botan::PKCS8::PEM_encode(private_key);
    priv_out.close();

    std::ofstream pub_out(key_folder + "/public.key", std::ios::binary);
    pub_out << Botan::X509::PEM_encode(private_key);
    pub_out.close();

    std::cout << "Ключи успешно сгенерированы в папке: " << key_folder << std::endl;
}

void hash_file_sha3(const std::string& input_file, const std::string& output_file) {
    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Ошибка открытия файла для хэширования: " << input_file << std::endl;
        return;
    }
    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto hash = Botan::HashFunction::create("SHA-3(256)");
    if (!hash) {
        std::cerr << "Ошибка создания SHA-3 хэш-функции." << std::endl;
        return;
    }
    hash->update(file_data);
    auto digest = hash->final();

    std::ofstream out(output_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(digest.data()), digest.size());
    out.close();

    std::cout << "Хэш файла сохранён в: " << output_file << std::endl;
}

void sign_file_gost(const std::string& input_file, const std::string& output_file, const std::string& private_key_file) {
    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Ошибка открытия файла для подписи: " << input_file << std::endl;
        return;
    }
    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto hash = Botan::HashFunction::create("Streebog-256");
    if (!hash) {
        std::cerr << "Ошибка создания ГОСТ хэш-функции." << std::endl;
        return;
    }
    hash->update(file_data);
    auto file_hash = hash->final();

    Botan::DataSource_Stream in_key(private_key_file);
    auto priv_key = Botan::PKCS8::load_key(in_key);

    Botan::AutoSeeded_RNG rng;
    Botan::PK_Signer signer(*priv_key, rng, "Raw");
    auto signature = signer.sign_message(file_hash, rng);

    std::ofstream out(output_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(signature.data()), signature.size());
    out.close();

    std::cout << "Файл успешно подписан, подпись сохранена в: " << output_file << std::endl;
}

void verify_file_hash(const std::string& input_file, const std::string& hash_file) {
    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Ошибка открытия файла: " << input_file << std::endl;
        return;
    }
    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto hash = Botan::HashFunction::create("SHA-3(256)");
    if (!hash) {
        std::cerr << "Ошибка создания SHA-3 хэш-функции." << std::endl;
        return;
    }
    hash->update(file_data);
    auto calculated_hash = hash->final();

    std::ifstream hash_in(hash_file, std::ios::binary);
    if (!hash_in) {
        std::cerr << "Ошибка открытия файла с хэш-суммой: " << hash_file << std::endl;
        return;
    }
    std::vector<uint8_t> stored_hash((std::istreambuf_iterator<char>(hash_in)), std::istreambuf_iterator<char>());
    hash_in.close();

    if (std::equal(calculated_hash.begin(), calculated_hash.end(), stored_hash.begin(), stored_hash.end())) {
        std::cout << "Хэш-суммы совпадают." << std::endl;
    }
    else {
        std::cerr << "Хэш-суммы не совпадают!" << std::endl;
    }
}

void verify_signature_gost(const std::string& input_file, const std::string& signature_file, const std::string& public_key_file) {
    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Ошибка открытия файла для проверки: " << input_file << std::endl;
        return;
    }
    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto hash = Botan::HashFunction::create("Streebog-256");
    if (!hash) {
        std::cerr << "Ошибка создания ГОСТ хэш-функции." << std::endl;
        return;
    }
    hash->update(file_data);
    auto file_hash = hash->final();

    std::ifstream sig_in(signature_file, std::ios::binary);
    if (!sig_in) {
        std::cerr << "Ошибка открытия файла подписи: " << signature_file << std::endl;
        return;
    }
    std::vector<uint8_t> signature((std::istreambuf_iterator<char>(sig_in)), std::istreambuf_iterator<char>());
    sig_in.close();

    Botan::DataSource_Stream pub_key_src(public_key_file);
    auto pub_key = Botan::X509::load_key(pub_key_src);

    Botan::PK_Verifier verifier(*pub_key, "Raw");

    bool valid = verifier.verify_message(file_hash, signature);

    if (valid) {
        std::cout << "Подпись действительна." << std::endl;
    }
    else {
        std::cerr << "Подпись НЕДЕЙСТВИТЕЛЬНА!" << std::endl;
    }
}

int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    std::string folder_path;
    while (true) {
        std::cout << "Введите путь к папке с текстовыми файлами: ";
        std::getline(std::cin, folder_path);

        if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
            break;
        }
        else {
            std::cerr << "Ошибка: Папка не существует или путь некорректен. Попробуйте снова.\n";
        }
    }

    std::string keys_folder;
    while (true) {
        std::cout << "Введите путь к папке с ключами: ";
        std::getline(std::cin, keys_folder);

        if (fs::exists(keys_folder) && fs::is_directory(keys_folder)) {
            std::cout << "Найдены существующие ключи! Использую их.\n";
            break;
        }
        else if (!fs::exists(keys_folder)) {
            std::cout << "Папка для ключей не существует. Пытаюсь создать...\n";
            try {
                fs::create_directories(keys_folder);
                std::cout << "Папка создана. Генерирую новые ключи...\n";
                generate_keys(keys_folder);
                break;
            }
            catch (const std::exception& e) {
                std::cerr << "Не удалось создать папку: " << e.what() << "\n";
            }
        }
        else {
            std::cerr << "Ошибка: Введённый путь не является папкой. Попробуйте снова.\n";
        }
    }

    while (true) {
        std::cout << "\nВыберите:\n";
        std::cout << "1 - Сформировать хэш файла (SHA-3)\n";
        std::cout << "2 - Подписать файл (ГОСТ Р 34.10.2012)\n";
        std::cout << "3 - Проверить хэш файла (SHA-3)\n";
        std::cout << "4 - Верифицировать ЭЦП (ГОСТ Р 34.10.2012)\n";
        std::cout << "0 - Выход\n";
        std::cout << "Введите ваш выбор: ";

        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Ошибка ввода! Пожалуйста, выберите 0-4\n";
            continue;
        }

        std::cin.ignore();

        if (choice == 0) {
            std::cout << "Выход из программы." << std::endl;
            break;
        }
        else if (choice == 1) {
            std::string in_filename = "in.txt";
            std::string out_filename = "hash.bin";
            std::cout << "Введите имя файла для хэширования (по умолчанию in.txt): ";
            std::getline(std::cin, in_filename);
            if (in_filename.empty()) in_filename = "in.txt";

            std::cout << "Введите имя выходного файла для хэша (по умолчанию hash.bin): ";
            std::getline(std::cin, out_filename);
            if (out_filename.empty()) out_filename = "hash.bin";

            hash_file_sha3(folder_path + "/" + in_filename, folder_path + "/" + out_filename);
        }
        else if (choice == 2) {
            std::string in_filename = "in.txt";
            std::string out_filename = "signature.bin";
            std::cout << "Введите имя файла для подписи (по умолчанию in.txt): ";
            std::getline(std::cin, in_filename);
            if (in_filename.empty()) in_filename = "in.txt";

            std::cout << "Введите имя файла для сохранения подписи (по умолчанию signature.bin): ";
            std::getline(std::cin, out_filename);
            if (out_filename.empty()) out_filename = "signature.bin";

            sign_file_gost(folder_path + "/" + in_filename, folder_path + "/" + out_filename, keys_folder + "/private.key");
        }
        else if (choice == 3) {
            std::string in_filename = "in.txt";
            std::string hash_filename = "hash.bin";
            std::cout << "Введите имя файла для проверки хэша (по умолчанию in.txt): ";
            std::getline(std::cin, in_filename);
            if (in_filename.empty()) in_filename = "in.txt";

            std::cout << "Введите имя файла с хэш-суммой (по умолчанию hash.bin): ";
            std::getline(std::cin, hash_filename);
            if (hash_filename.empty()) hash_filename = "hash.bin";

            verify_file_hash(folder_path + "/" + in_filename, folder_path + "/" + hash_filename);
        }
        else if (choice == 4) {
            std::string in_filename = "in.txt";
            std::string sig_filename = "signature.bin";
            std::cout << "Введите имя файла для проверки подписи (по умолчанию in.txt): ";
            std::getline(std::cin, in_filename);
            if (in_filename.empty()) in_filename = "in.txt";

            std::cout << "Введите имя файла с подписью (по умолчанию signature.bin): ";
            std::getline(std::cin, sig_filename);
            if (sig_filename.empty()) sig_filename = "signature.bin";

            verify_signature_gost(folder_path + "/" + in_filename, folder_path + "/" + sig_filename, keys_folder + "/public.key");
        }
        else {
            std::cout << "Некорректный выбор. Попробуйте снова.\n";
        }
    }

    return 0;
}
