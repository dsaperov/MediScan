# Описание датасетов для задачи определения границ новообразования и их характеристик:
1. [ISIC2018_Task1-2_Training_Input](https://disk.yandex.ru/d/ltui3Ovrlww-vQ) - обучающая выборка содержит фотографии новообразованиий.
2. [ISIC2018_Task1_Training_GroundTruth](https://disk.yandex.ru/d/MCnY7SCJnhg1gA) - результат обработки обучающей выборки, содержит сегментированные фотографии, показывающие области нахождения новообразований, другими словами - определены границы новообразований.
3. [ISIC2018_Task2_Training_GroundTruth_v3](https://disk.yandex.ru/d/zE38pqePiPmIfw) - результат обработки обучающей выборки, содержит 5 масок ответа на каждую фотографию.
4. [ISIC2018_Task1-2_Validation_Input](https://disk.yandex.ru/d/EKiA9nceuM4pAw) - валидационная выборка содержит изображения, на основе данного датасета подбираются оптимальные наборы гиперпараметров.
5. [ISIC2018_Task1_Validation_GroundTruth](https://disk.yandex.ru/d/wGyOmQkx-zHZIQ) - результат обработки валидационной выборки, содержит сегментированные фотографии, показывающие области нахождения новообразований, другими словами - определены границы новообразований.
6. [ISIC2018_Task2_Validation_GroundTruth](https://disk.yandex.ru/d/_Q-rzyp30evQHA) - результат обработки валидационной выборки, содержит 5 масок ответа на каждую фотографию.
7. [ISIC2018_Task1-2_Test_Input](https://disk.yandex.ru/d/pPbsLaXIyQgsFg) - тестовая выборка, не используется в обучении, нужна для тестирования модели.
8. [ISIC2018_Task1_Test_GroundTruth](https://disk.yandex.ru/d/qWN8-Co4mhmK5Q) - результат обработки тестовой выборки, содержит сегментированные фотографии, показывающие области нахождения новообразований, другими словами - определены границы новообразований.
9. [ISIC2018_Task2_Test_GroundTruth](https://disk.yandex.ru/d/AUUaFACGLnimWA) - результат обработки тестовой выборки, содержит 5 масок ответа на каждую фотографию.

# Описание датасетов для задачи классификации:
1. [ISIC2018_Task3_Training_Input](https://disk.yandex.ru/d/T3-12lvh5bhJCQ) - обучающая выборка содержит фотографии новообразованиий.
2. [ISIC2018_Task3_Training_GroundTruth](https://disk.yandex.ru/d/0rgFF8TyHB38GA) - содержит CSV файл с классификацией новообразований по типу. (1 \ 0 в столбце с типом новообразования)
   * MEL - melanoma
   * NV - melanocytic nevi
   * BCC - basal cell carcinoma
   * AKIEC - actinic keratosis intraepithelial carcinoma
   * BKL - benign keratosis
   * DF - dermatofibroma
   * VASC - vascular lesion
   * Quantities:
     - MEL: 1113	
     - NV: 6705	
     - BCC: 514	
     - AKIEC: 327	
     - BKL: 1099	
     - DF: 115	
     - VASC: 142
3. [ISIC2018_Task3_Validation_Input](https://disk.yandex.ru/d/GZ0sgy9be5bY_w) - валидационная выборка содержит изображения, на основе данного датасета подбираются оптимальные наборы гиперпараметров.
4. [ISIC2018_Task3_Validation_GroundTruth](https://disk.yandex.ru/d/izblELCRRG2QWQ) - результат обработки валидационной выборки, содержит CSV файл с классификацией новообразований по типу. (1 \ 0 в столбце с типом новообразования)
5. [ISIC2018_Task3_Test_Input](https://disk.yandex.ru/d/rVU1NiiFezEWxA) - тестовая выборка, не используется в обучении, нужна для тестирования модели.
6. [ISIC2018_Task3_Test_GroundTruth](https://disk.yandex.ru/d/M9j1vuRcW_F-ig) - результат обработки тестовой выборки, содержит CSV файл с классификацией новообразований по типу. (1 \ 0 в столбце с типом новообразования)