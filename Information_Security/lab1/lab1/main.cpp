#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>

#define REG_PATH L"SOFTWARE\\MySecureApp"
#define REG_VALUE L"SerialNumber"

std::string GetSerialNumber(HANDLE hDevice) {
    STORAGE_PROPERTY_QUERY storageQuery = { 0 };
    storageQuery.PropertyId = StorageDeviceProperty;
    storageQuery.QueryType = PropertyStandardQuery;

    BYTE outputBuffer[512] = { 0 };
    DWORD bytesReturned = 0;

    BOOL success = DeviceIoControl(
        hDevice,
        IOCTL_STORAGE_QUERY_PROPERTY,
        &storageQuery, sizeof(STORAGE_PROPERTY_QUERY),
        &outputBuffer, sizeof(outputBuffer),
        &bytesReturned,
        NULL
    );

    if (!success) {
        std::cerr << "DeviceIoControl failed with error: " << GetLastError() << std::endl;
        return "";
    }

    STORAGE_DEVICE_DESCRIPTOR* deviceDescriptor = (STORAGE_DEVICE_DESCRIPTOR*)outputBuffer;
    DWORD serialOffset = deviceDescriptor->SerialNumberOffset;

    if (serialOffset == 0) {
        std::cerr << "Serial number not found." << std::endl;
        return "";
    }

    std::string serial(reinterpret_cast<char*>(outputBuffer + serialOffset));
    return serial;
}

std::string GetSerialFromRegistry() {
    HKEY hKey;
    LONG result = RegOpenKeyExW(HKEY_CURRENT_USER, REG_PATH, 0, KEY_READ, &hKey);

    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to open registry key. Error: " << result << std::endl;
        return "";
    }

    char serialBuffer[256] = { 0 };
    DWORD bufferSize = sizeof(serialBuffer);
    result = RegQueryValueExW(hKey, REG_VALUE, NULL, NULL, (LPBYTE)serialBuffer, &bufferSize);
    RegCloseKey(hKey);

    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to read from registry. Error: " << result << std::endl;
        return "";
    }

    return std::string(serialBuffer);
}

int main() {
    HANDLE hDevice = CreateFileW(
        L"\\\\.\\PhysicalDrive0",
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );

    if (hDevice == INVALID_HANDLE_VALUE) {
        std::cerr << "Cannot open the driver. Error: " << GetLastError() << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();
        return 1;
    }

    std::string currentSerial = GetSerialNumber(hDevice);
    CloseHandle(hDevice);

    if (currentSerial.empty()) {
        std::cerr << "Failed to retrieve serial number." << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();
        return 1;
    }

    std::string storedSerial = GetSerialFromRegistry();

    if (storedSerial.empty()) {
        std::cerr << "No activation found!" << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();
        return 1;
    }

    if (storedSerial == currentSerial) {
        std::cout << "The program is activated!!!" << std::endl;
    }
    else {
        std::cout << "No access!!! Try again." << std::endl;
    }

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}
