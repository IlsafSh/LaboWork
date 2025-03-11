#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>

#define PROTECTED_PROGRAM_PATH "lab1.exe"

// √лобальна€ переменна€ в секции .data (зарезервированное место дл€ серийного номера)
#pragma section(".data", read, write)
__declspec(allocate(".data")) char modelHDD[64] = "###############################################################";

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

    return std::string(reinterpret_cast<char*>(buffer) + desc->SerialNumberOffset);
}

int main() {
    std::string currentSerial = getHddSerial();
    if (currentSerial.empty()) {
        std::cout << "Error: Unable to retrieve HDD serial number!" << std::endl;
        system("pause");
        return 1;
    }

    // ѕреобразуем модель HDD из char* в std::string, чтобы можно было использовать find()
    std::string modelHDDString(modelHDD);

    if ((modelHDDString.find(currentSerial) != std::string::npos) && (modelHDDString != "")) {
        std::cout << "The program is activated!" << std::endl;
    }
    else {
        std::cout << "No access! Try again." << std::endl;
    }

    system("pause");
    return 0;
}