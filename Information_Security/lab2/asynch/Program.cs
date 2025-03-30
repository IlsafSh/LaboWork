using System;
using System.IO;
using System.Text;
using Org.BouncyCastle.Crypto;
using Org.BouncyCastle.Crypto.Encodings;
using Org.BouncyCastle.Crypto.Engines;
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.OpenSsl;
using System.Diagnostics;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.WriteLine("------ АСИММЕТРИЧНОЕ ШИФРОВАНИЕ ------");
            Console.WriteLine("- Bouncy Castle ---- RSA ---- 2048 bit -");

            string keyFolder, textFileFolder;

            if (args.Length >= 2)
            {
                keyFolder = args[0];
                textFileFolder = args[1];

                if (!Directory.Exists(keyFolder) || !Directory.Exists(textFileFolder))
                {
                    Console.WriteLine("Ошибка: Один или оба указанных пути не существуют");
                    return;
                }
            }
            else
            {
                Console.Write("Введите путь к папке для ключей (например, C:\\Keys): ");
                keyFolder = GetValidPath();
                Console.Write("Введите путь к папке для текстовых файлов (например, C:\\Files): ");
                textFileFolder = GetValidPath();
            }

            string privateKeyPath = Path.Combine(keyFolder, "private_key.pem");
            string publicKeyPath = Path.Combine(keyFolder, "public_key.pem");

            AsymmetricCipherKeyPair keyPair = null;

            if (File.Exists(privateKeyPath) && File.Exists(publicKeyPath))
            {
                Console.WriteLine("Найдены существующие ключи! Использую их");
            }
            else
            {
                Console.WriteLine("Ключи не найдены, генерирую новые...");
                keyPair = GenerateKeyPair(2048);
                File.WriteAllText(privateKeyPath, ExportPrivateKey(keyPair));
                File.WriteAllText(publicKeyPath, ExportPublicKey(keyPair));
                Console.WriteLine("Ключи сохранены");
            }

            while (true)
            {
                Console.WriteLine("\nВыберите действие:");
                Console.WriteLine("1 - Зашифровать файл");
                Console.WriteLine("2 - Расшифровать файл");
                Console.WriteLine("0 - Выход");
                Console.Write("...");

                string choice = Console.ReadLine();

                if (choice == "1")
                {
                    // Входной файл: если не указан, то используем in.txt по умолчанию
                    Console.Write("Введите имя входного файла (по умолчанию in.txt): ");
                    string inputFile = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(inputFile))
                        inputFile = "in.txt";  // Файл по умолчанию
                    inputFile = Path.Combine(textFileFolder, inputFile);

                    // Выходной зашифрованный файл: если не указан, то используем out_enc.txt по умолчанию
                    Console.Write("Введите имя зашифрованного файла (по умолчанию out_enc.txt): ");
                    string encryptedFile = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(encryptedFile))
                        encryptedFile = "out_enc.txt";  // Файл по умолчанию
                    encryptedFile = Path.Combine(textFileFolder, encryptedFile);

                    RsaKeyParameters publicKeyParam = LoadPublicKey(File.ReadAllText(publicKeyPath));

                    var stopwatch = Stopwatch.StartNew();
                    EncryptFile(inputFile, encryptedFile, publicKeyParam);
                    stopwatch.Stop();

                    Console.WriteLine($"Файл {inputFile} зашифрован в {encryptedFile} за {stopwatch.ElapsedMilliseconds} мс.");
                }
                else if (choice == "2")
                {
                    // Зашифрованный файл: если не указан, то используем out_enc.txt по умолчанию
                    Console.Write("Введите имя зашифрованного файла (по умолчанию out_enc.txt): ");
                    string encryptedFile = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(encryptedFile))
                        encryptedFile = "out_enc.txt";  // Файл по умолчанию
                    encryptedFile = Path.Combine(textFileFolder, encryptedFile);

                    // Расшифрованный файл: если не указан, то используем out_dec.txt по умолчанию
                    Console.Write("Введите имя расшифрованного файла (по умолчанию out_dec.txt): ");
                    string decryptedFile = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(decryptedFile))
                        decryptedFile = "out_dec.txt";  // Файл по умолчанию
                    decryptedFile = Path.Combine(textFileFolder, decryptedFile);

                    AsymmetricKeyParameter privateKeyParam = LoadPrivateKey(File.ReadAllText(privateKeyPath));

                    var stopwatch = Stopwatch.StartNew();
                    DecryptFile(encryptedFile, decryptedFile, privateKeyParam);
                    stopwatch.Stop();

                    Console.WriteLine($"Файл {encryptedFile} расшифрован в {decryptedFile} за {stopwatch.ElapsedMilliseconds} мс.");

                    Console.WriteLine("\nСодержимое расшифрованного файла:");
                    try
                    {
                        string decryptedContent = File.ReadAllText(decryptedFile, Encoding.UTF8);
                        Console.WriteLine(decryptedContent);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Ошибка при чтении файла: " + ex.Message);
                    }
                }
                else if (choice == "0")
                {
                    Console.WriteLine("Работа программы завершена");
                    break;
                }
                else
                {
                    Console.WriteLine("Ошибка: неверный ввод, попробуйте снова");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error: " + ex.Message);
        }
    }

    private static string GetValidPath()
    {
        while (true)
        {
            string path = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(path) && Directory.Exists(path))
                return path;

            Console.Write("Ошибка: путь не существует. Повторите ввод: ");
        }
    }

    public static AsymmetricCipherKeyPair GenerateKeyPair(int keySize)
    {
        var generator = new RsaKeyPairGenerator();
        generator.Init(new KeyGenerationParameters(new Org.BouncyCastle.Security.SecureRandom(), keySize));
        return generator.GenerateKeyPair();
    }

    public static string ExportPrivateKey(AsymmetricCipherKeyPair keyPair)
    {
        using (StringWriter stringWriter = new StringWriter())
        {
            PemWriter pemWriter = new PemWriter(stringWriter);
            pemWriter.WriteObject(keyPair.Private);
            return stringWriter.ToString();
        }
    }

    public static string ExportPublicKey(AsymmetricCipherKeyPair keyPair)
    {
        using (StringWriter stringWriter = new StringWriter())
        {
            PemWriter pemWriter = new PemWriter(stringWriter);
            pemWriter.WriteObject(keyPair.Public);
            return stringWriter.ToString();
        }
    }

    public static AsymmetricKeyParameter LoadPrivateKey(string privateKeyPem)
    {
        using (StringReader stringReader = new StringReader(privateKeyPem))
        {
            PemReader pemReader = new PemReader(stringReader);
            object keyObject = pemReader.ReadObject();

            if (keyObject is AsymmetricCipherKeyPair keyPair)
            {
                return keyPair.Private;
            }
            else if (keyObject is AsymmetricKeyParameter keyParameter)
            {
                return keyParameter;
            }
            else
            {
                throw new Exception("Ошибка: Невозможно распознать закрытый ключ.");
            }
        }
    }

    public static RsaKeyParameters LoadPublicKey(string publicKeyPem)
    {
        using (StringReader stringReader = new StringReader(publicKeyPem))
        {
            PemReader pemReader = new PemReader(stringReader);
            return (RsaKeyParameters)pemReader.ReadObject();
        }
    }

    public static void EncryptFile(string inputFile, string outputFile, RsaKeyParameters publicKey)
    {
        byte[] inputData = File.ReadAllBytes(inputFile);

        var encryptEngine = new Pkcs1Encoding(new RsaEngine());
        encryptEngine.Init(true, publicKey);

        byte[] encryptedData = encryptEngine.ProcessBlock(inputData, 0, inputData.Length);
        File.WriteAllBytes(outputFile, encryptedData);
    }

    public static void DecryptFile(string inputFile, string outputFile, AsymmetricKeyParameter privateKey)
    {
        byte[] encryptedData = File.ReadAllBytes(inputFile);

        var decryptEngine = new Pkcs1Encoding(new RsaEngine());
        decryptEngine.Init(false, privateKey);

        byte[] decryptedData = decryptEngine.ProcessBlock(encryptedData, 0, encryptedData.Length);
        File.WriteAllBytes(outputFile, decryptedData);
    }
}
