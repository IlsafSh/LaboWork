#include <Windows.h>
#include <winioctl.h>
#include <iostream>
#include <string>

#define SERIAL_OFFSET 0x200  // Смещение для чтения серийного номера

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

std::string readStoredSerial()
{
    FILE* file;
    errno_t err = fopen_s(&file, "lab1.exe", "rb");
    if (err != 0 || file == nullptr)
    {
        std::cerr << "Error: Unable to open executable!" << std::endl;
        return "";
    }

    fseek(file, SERIAL_OFFSET, SEEK_SET);
    char buffer[32] = { 0 };
    fread(buffer, 1, sizeof(buffer), file);
    fclose(file);

    return std::string(buffer);
}

int main()
{
    std::string storedSerial = readStoredSerial();
    std::string currentSerial = getHDDSerial();

    if (storedSerial.empty() || currentSerial.empty())
    {
        std::cerr << "Error: Could not retrieve serial numbers!" << std::endl;
        return 1;
    }

    if (storedSerial == currentSerial)
    {
        std::cout << "The program is activated!" << std::endl;
    }
    else
    {
        std::cout << "No access! Try again." << std::endl;
    }

    system("pause");
    return 0;
}
