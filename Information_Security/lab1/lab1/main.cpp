#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>

#define PATCH_OFFSET 0x05000 // Смещение для чтения серийного номера
#define PROTECTED_PROGRAM_PATH "lab1.exe"

std::string getHddSerial() {
    HANDLE hDevice = CreateFileW(L"\\\\.\\PhysicalDrive0", GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);

    if (hDevice == INVALID_HANDLE_VALUE) {
        std::wcout << L"Cannot open the drive! Error: " << GetLastError() << std::endl;
        return "";
    }

    STORAGE_PROPERTY_QUERY query = { StorageDeviceProperty, PropertyStandardQuery };
    BYTE buffer[512] = { 0 };
    DWORD dwOutBytes = 0;

    BOOL success = DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
        buffer, sizeof(buffer), &dwOutBytes, NULL);

    CloseHandle(hDevice);

    if (!success) {
        std::wcout << L"DeviceIoControl failed with error: " << GetLastError() << std::endl;
        return "";
    }

    STORAGE_DEVICE_DESCRIPTOR* desc = reinterpret_cast<STORAGE_DEVICE_DESCRIPTOR*>(buffer);
    if (desc->SerialNumberOffset == 0) {
        std::wcout << L"Error: No serial number found!" << std::endl;
        return "";
    }

    std::string serial(reinterpret_cast<char*>(buffer) + desc->SerialNumberOffset);
    return serial;
}


int main() {
    std::string currentSerial = getHddSerial();
    if (currentSerial.empty()) {
        std::cout << "Error: Unable to retrieve HDD serial number!" << std::endl;
        system("pause");
        return 1;
    }

    // Открываем саму программу для проверки серийного номера
    FILE* file;
    fopen_s(&file, PROTECTED_PROGRAM_PATH, "rb");
    if (!file) {
        std::cout << "Error: Unable to open lab1.exe!" << std::endl;
        return 1;
    }

    fseek(file, PATCH_OFFSET, SEEK_SET);
    char storedSerial[64] = { 0 };
    fread(storedSerial, sizeof(char), sizeof(storedSerial) - 1, file);
    fclose(file);

    if (currentSerial == std::string(storedSerial)) {
        std::cout << "The program is activated!" << std::endl;
    }
    else {
        std::cout << "No access! Try again." << std::endl;
    }

    system("pause");
    return 0;
}
