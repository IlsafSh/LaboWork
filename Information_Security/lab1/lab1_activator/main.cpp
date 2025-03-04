#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>

#define SERIAL_OFFSET 0x200  // Смещение для записи серийного номера
#define PROTECTED_PROGRAM_PATH "../lab1/x64/Release/lab1.exe"

std::string getHDDSerial()
{
    STORAGE_PROPERTY_QUERY query = { StorageDeviceProperty, PropertyStandardQuery };
    STORAGE_DESCRIPTOR_HEADER header = { 0 };
    DWORD returned = 0;

    HANDLE hDevice = CreateFileW(L"\\\\.\\PhysicalDrive0", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hDevice == INVALID_HANDLE_VALUE)
    {
        std::cerr << "Error: Cannot open drive!" << std::endl;
        return "";
    }

    if (!DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &header, sizeof(header), &returned, nullptr))
    {
        std::cerr << "Error: Query failed!" << std::endl;
        CloseHandle(hDevice);
        return "";
    }

    BYTE* buffer = new BYTE[header.Size];
    if (!DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), buffer, header.Size, &returned, nullptr))
    {
        std::cerr << "Error: Query failed!" << std::endl;
        delete[] buffer;
        CloseHandle(hDevice);
        return "";
    }

    STORAGE_DEVICE_DESCRIPTOR* descriptor = (STORAGE_DEVICE_DESCRIPTOR*)buffer;
    std::string serial = (descriptor->SerialNumberOffset ? (char*)(buffer + descriptor->SerialNumberOffset) : "UNKNOWN");

    delete[] buffer;
    CloseHandle(hDevice);
    return serial;
}

int main()
{
    std::string serial = getHDDSerial();
    if (serial.empty())
    {
        std::cerr << "Error: Could not retrieve serial number!" << std::endl;
        return 1;
    }

    FILE* file;
    errno_t err = fopen_s(&file, PROTECTED_PROGRAM_PATH, "rb+");
    if (err != 0 || file == nullptr)
    {
        std::cerr << "Error: Unable to open " << PROTECTED_PROGRAM_PATH << "!" << std::endl;
        return 1;
    }

    fseek(file, SERIAL_OFFSET, SEEK_SET);
    fwrite(serial.c_str(), 1, serial.size(), file);
    fclose(file);

    std::cout << "Activation successful!" << std::endl;
    system("pause");
    return 0;
}