#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>

#define PATCH_OFFSET 0x05000 // Смещение для записи серийного номера
#define PROTECTED_PROGRAM_PATH "lab1.exe"

// Функция для получения серийного номера HDD через DeviceIoControl
std::string getHddSerial() {
    HANDLE hDevice = CreateFileW(L"\\\\.\\PhysicalDrive0", GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);

    if (hDevice == INVALID_HANDLE_VALUE) {
        std::cerr << "Cannot open the drive! Error: " << GetLastError() << std::endl;
        return "";
    }

    STORAGE_PROPERTY_QUERY query = { StorageDeviceProperty, PropertyStandardQuery };
    BYTE buffer[512] = { 0 };
    DWORD dwOutBytes = 0;

    BOOL success = DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
        buffer, sizeof(buffer), &dwOutBytes, NULL);

    CloseHandle(hDevice);

    if (!success) {
        std::cerr << "DeviceIoControl failed with error: " << GetLastError() << std::endl;
        return "";
    }

    STORAGE_DEVICE_DESCRIPTOR* desc = reinterpret_cast<STORAGE_DEVICE_DESCRIPTOR*>(buffer);

    // Проверяем, есть ли серийный номер
    if (desc->SerialNumberOffset == 0 || desc->SerialNumberOffset >= sizeof(buffer)) {
        std::cerr << "Error: No valid serial number found!" << std::endl;
        return "";
    }

    // Читаем серийный номер и убираем возможные мусорные символы
    std::string serial(reinterpret_cast<char*>(buffer) + desc->SerialNumberOffset);
    serial.erase(std::remove(serial.begin(), serial.end(), ' '), serial.end()); // Убираем пробелы

    return serial;
}

int main() {
    // Получаем текущий серийный номер HDD
    std::string newSerial = getHddSerial();
    if (newSerial.empty()) {
        std::cout << "Error: Unable to retrieve HDD serial number!" << std::endl;
        system("pause");
        return 1;
    }

    // Открываем lab1.exe для чтения и записи
    FILE* file;
    fopen_s(&file, PROTECTED_PROGRAM_PATH, "r+b");
    if (!file) {
        std::cerr << "Error: Unable to open lab1.exe!" << std::endl;
        return 1;
    }

    // Читаем текущий серийный номер
    fseek(file, PATCH_OFFSET, SEEK_SET);
    char storedSerial[64] = { 0 };
    fread(storedSerial, sizeof(char), sizeof(storedSerial) - 1, file);

    // Если серийный номер совпадает, ничего не меняем
    if (newSerial == std::string(storedSerial)) {
        std::cout << "Program is already activated on this device!" << std::endl;
        fclose(file);
        system("pause");
        return 0;
    }

    // Стираем старый серийный номер
    fseek(file, PATCH_OFFSET, SEEK_SET);
    char emptyData[64] = { 0 }; // Заполняем массив нулями
    fwrite(emptyData, sizeof(char), sizeof(emptyData), file);
    fflush(file);

    // Записываем новый серийный номер
    fseek(file, PATCH_OFFSET, SEEK_SET);
    fwrite(newSerial.c_str(), sizeof(char), newSerial.size(), file);
    fclose(file);

    std::cout << "Activation successful!" << std::endl;
    system("pause");
    return 0;
}