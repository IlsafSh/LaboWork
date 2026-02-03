#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>
#include <shlwapi.h>  // Для PathRemoveFileSpec
#pragma comment(lib, "Shlwapi.lib")

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
    std::string newSerial = getHddSerial();
    if (newSerial.empty()) {
        std::cout << "Error: Unable to retrieve HDD serial number!" << std::endl;
        system("pause");
        return 1;
    }

    char activatorPath[MAX_PATH];
    GetModuleFileNameA(NULL, activatorPath, MAX_PATH);
    PathRemoveFileSpecA(activatorPath);
    std::string protectedExePath = "lab1.exe";

    FILE* file;
    fopen_s(&file, protectedExePath.c_str(), "r+b");

    if (!file) {
        std::cerr << "Error: Unable to open " << protectedExePath << "!" << std::endl;
        return 1;
    }

    // Поиск адреса modelHDD в бинарном файле
    const char marker[64] = "###############################################################";
    char buffer[1024];

    size_t offset = 0;
    while (fread(buffer, 1, sizeof(buffer), file)) {
        for (size_t i = 0; i < sizeof(buffer) - sizeof(marker); ++i) {
            if (memcmp(&buffer[i], marker, sizeof(marker)) == 0) {
                offset += i;
                goto found;
            }
        }
        offset += sizeof(buffer);
    }

    std::cerr << "Error: marker of modelHDD not found in " << protectedExePath << "!" << std::endl;
    system("pause");
    fclose(file);
    return 1;

found:
    fseek(file, offset, SEEK_SET);
    fwrite(newSerial.c_str(), sizeof(char), newSerial.size(), file);
    fclose(file);

    std::cout << "Activation successful!" << std::endl;
    system("pause");
    return 0;
}