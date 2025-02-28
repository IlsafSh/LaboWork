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

bool SaveSerialToRegistry(const std::string& serial) {
    HKEY hKey;
    LONG result = RegCreateKeyExW(HKEY_CURRENT_USER, REG_PATH, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hKey, NULL);

    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to create/open registry key. Error: " << result << std::endl;
        return false;
    }

    result = RegSetValueExW(hKey, REG_VALUE, 0, REG_SZ, (const BYTE*)serial.c_str(), serial.size() + 1);
    RegCloseKey(hKey);

    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to write to registry. Error: " << result << std::endl;
        return false;
    }

    return true;
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

    std::string serial = GetSerialNumber(hDevice);
    CloseHandle(hDevice);

    if (serial.empty()) {
        std::cerr << "Failed to retrieve serial number." << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();
        return 1;
    }

    if (SaveSerialToRegistry(serial)) {
        std::cout << "Activation Successful!" << std::endl;
    }
    else {
        std::cout << "Activation Failed!" << std::endl;
    }

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}