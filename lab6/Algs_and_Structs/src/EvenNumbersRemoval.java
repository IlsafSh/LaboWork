import java.util.ArrayList;
import java.util.Random;

public class EvenNumbersRemoval {
    public static void main(String[] args) {
        // Задаем размеры массивов
        int[] sizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};

        // Таблица для результатов
        System.out.println();
        System.out.printf("%-12s | %-15s%n", "Размер массива", "Время (нс)");
        System.out.println("-------------------------------");

        // Прогрев JVM
        ArrayList<Integer> array_w = generateRandomArray(10);
            for (int i = 0; i < 5; i++) {
                removeEvenNumbers(array_w);
            }
        System.out.println(array_w.toString());

        for (int size : sizes) {
            // Генерация случайного массива
            ArrayList<Integer> array = generateRandomArray(size);

            // Вывод исходного состояния массива (для первых 20 элементов)
            System.out.println("Исходный массив (первые 20 элементов): " + previewArray(array, 20));

            // Засекаем время
            long startTime = System.nanoTime();
            removeEvenNumbers(array);
            long endTime = System.nanoTime();

            // Вывод состояния массива после удаления чётных чисел
            System.out.println("Массив после удаления чётных чисел (первые 20 элементов): " + previewArray(array, 20));

            // Вывод результата
            System.out.printf("%-12d | %-15d%n", size, (endTime - startTime));
        }
    }

    /**
     * Метод для генерации случайного массива заданного размера
     */
    public static ArrayList<Integer> generateRandomArray(int size) {
        ArrayList<Integer> array = new ArrayList<>(size);
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            array.add(random.nextInt(1_000_000)); // Генерируем числа от 0 до 999999
        }
        return array;
    }

    /**
     * Метод для удаления четных чисел из массива
     */
    public static void removeEvenNumbers(ArrayList<Integer> array) {
        array.removeIf(number -> number % 2 == 0); // Удаление чётных чисел
    }

    /**
     * Метод для получения первых N элементов массива в виде строки
     */
    public static String previewArray(ArrayList<Integer> array, int limit) {
        StringBuilder preview = new StringBuilder("[");
        for (int i = 0; i < Math.min(array.size(), limit); i++) {
            preview.append(array.get(i));
            if (i < Math.min(array.size(), limit) - 1) {
                preview.append(", ");
            }
        }
        if (array.size() > limit) {
            preview.append(", ..."); // Указание на то, что массив обрезан
        }
        preview.append("]");
        return preview.toString();
    }
}
