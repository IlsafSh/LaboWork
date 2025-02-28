#include <Windows.h>
#include <iostream>

#define REG_PATH L"SOFTWARE\\MySecureApp"

bool DeleteActivationData() {
    LONG result = RegDeleteKeyW(HKEY_CURRENT_USER, REG_PATH);

    if (result == ERROR_SUCCESS) {
        std::cout << "Activation data successfully removed!" << std::endl;
        return true;
    }
    else if (result == ERROR_FILE_NOT_FOUND) {
        std::cout << "No activation data found!" << std::endl;
        return false;
    }
    else {
        std::cerr << "Failed to remove activation data. Error: " << result << std::endl;
        return false;
    }
}

int main() {
    if (DeleteActivationData()) {
        std::cout << "Deactivation complete." << std::endl;
    }
    else {
        std::cerr << "Deactivation failed!" << std::endl;
    }

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}