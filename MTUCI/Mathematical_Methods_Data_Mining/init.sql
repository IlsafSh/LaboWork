-- MySQL dump 10.13  Distrib 8.0.45, for Linux (x86_64)
--
-- Host: localhost    Database: telecom_db
-- ------------------------------------------------------
-- Server version	8.0.45

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Calls`
--

DROP TABLE IF EXISTS `Calls`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Calls` (
  `call_ID` int NOT NULL AUTO_INCREMENT,
  `c_sub_ID` int NOT NULL,
  `c_city_ID` int NOT NULL,
  `c_date` datetime NOT NULL,
  `c_time_of_day` enum('день','ночь') COLLATE utf8mb4_unicode_ci NOT NULL,
  `c_duration` int NOT NULL,
  PRIMARY KEY (`call_ID`),
  KEY `c_sub_ID` (`c_sub_ID`),
  KEY `c_city_ID` (`c_city_ID`),
  CONSTRAINT `Calls_ibfk_1` FOREIGN KEY (`c_sub_ID`) REFERENCES `Subscribers` (`sub_ID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `Calls_ibfk_2` FOREIGN KEY (`c_city_ID`) REFERENCES `Cities` (`city_ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Calls`
--

LOCK TABLES `Calls` WRITE;
/*!40000 ALTER TABLE `Calls` DISABLE KEYS */;
INSERT INTO `Calls` VALUES (1,1,2,'2026-03-10 14:30:00','день',12),(2,1,4,'2026-03-11 02:15:00','ночь',45),(4,3,5,'2026-03-12 23:50:00','ночь',22),(5,4,1,'2026-03-13 16:40:00','день',8),(6,5,3,'2026-03-14 09:20:00','день',35);
/*!40000 ALTER TABLE `Calls` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Cities`
--

DROP TABLE IF EXISTS `Cities`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Cities` (
  `city_ID` int NOT NULL AUTO_INCREMENT,
  `city_name` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`city_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Cities`
--

LOCK TABLES `Cities` WRITE;
/*!40000 ALTER TABLE `Cities` DISABLE KEYS */;
INSERT INTO `Cities` VALUES (1,'Москва'),(2,'Санкт-Петербург'),(3,'Казань'),(4,'Новосибирск'),(5,'Екатеринбург');
/*!40000 ALTER TABLE `Cities` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Discounts`
--

DROP TABLE IF EXISTS `Discounts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Discounts` (
  `d_city_ID` int NOT NULL,
  `d_min_duration` int NOT NULL,
  `d_max_duration` int NOT NULL,
  `d_percent` decimal(5,2) NOT NULL,
  PRIMARY KEY (`d_city_ID`,`d_min_duration`),
  CONSTRAINT `Discounts_ibfk_1` FOREIGN KEY (`d_city_ID`) REFERENCES `Cities` (`city_ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Discounts`
--

LOCK TABLES `Discounts` WRITE;
/*!40000 ALTER TABLE `Discounts` DISABLE KEYS */;
INSERT INTO `Discounts` VALUES (1,10,30,5.00),(1,31,1000,10.00),(2,15,45,3.00),(2,46,1000,8.00),(3,20,60,7.00),(3,61,1000,12.00);
/*!40000 ALTER TABLE `Discounts` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Subscribers`
--

DROP TABLE IF EXISTS `Subscribers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Subscribers` (
  `sub_ID` int NOT NULL AUTO_INCREMENT,
  `sub_name` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  `sub_phone` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `sub_inn` varchar(12) COLLATE utf8mb4_unicode_ci NOT NULL,
  `sub_account` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`sub_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Subscribers`
--

LOCK TABLES `Subscribers` WRITE;
/*!40000 ALTER TABLE `Subscribers` DISABLE KEYS */;
INSERT INTO `Subscribers` VALUES (1,'ООО \"Альфа\"','8-495-999-99-99','7712345678','40702810123450000001'),(2,'ЗАО \"Бета-Торг\"','8-812-987-65-43','7898765432','40702810123450000002'),(3,'ПАО \"Гамма Строй\"','8-843-111-22-33','1655123456','40702810123450000003'),(4,'ИП Иванов А.А.','8-383-555-44-33','5401123456','40702810123450000004'),(5,'ООО \"Дельта-IT\"','8-343-777-88-99','6671123456','40702810123450000005');
/*!40000 ALTER TABLE `Subscribers` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Tariffs`
--

DROP TABLE IF EXISTS `Tariffs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Tariffs` (
  `t_city_ID` int NOT NULL,
  `t_time_of_day` enum('день','ночь') COLLATE utf8mb4_unicode_ci NOT NULL,
  `t_price` decimal(7,2) NOT NULL,
  PRIMARY KEY (`t_city_ID`,`t_time_of_day`),
  CONSTRAINT `Tariffs_ibfk_1` FOREIGN KEY (`t_city_ID`) REFERENCES `Cities` (`city_ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Tariffs`
--

LOCK TABLES `Tariffs` WRITE;
/*!40000 ALTER TABLE `Tariffs` DISABLE KEYS */;
INSERT INTO `Tariffs` VALUES (1,'день',6.00),(1,'ночь',3.00),(2,'день',4.80),(2,'ночь',2.50),(3,'день',6.00),(3,'ночь',3.50),(4,'день',7.50),(4,'ночь',4.00),(5,'день',6.80),(5,'ночь',3.80);
/*!40000 ALTER TABLE `Tariffs` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-03-13 23:08:21
