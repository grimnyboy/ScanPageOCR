import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import re

OCR_ENGINE = None

try:
    import easyocr
    OCR_ENGINE = "easyocr"
    print("Използва се EasyOCR")
except ImportError:
    try:
        import pytesseract
        OCR_ENGINE = "tesseract"
        print("Използва се Tesseract")
    except ImportError:
        print("ГРЕШКА: Няма наличен OCR engine!")
        print("Инсталирай: pip install easyocr")
        exit(1)


class DocumentOCR:
    def __init__(self, input_folder=".", output_folder="output", auto_split=False,
                 word_spacing_tolerance=80, left_word_penalty_threshold=150,
                 confidence_threshold=0.3, sharpening_strength=1.3):
        """
        auto_split: True = автоматично разделяне на две страници
                    False = третира всяко изображение като една страница
        
        CONFIGURABLE PARAMETERS:
        - word_spacing_tolerance: X-разстояние толеранция за групиране на думи (px)
        - left_word_penalty_threshold: Наказание за думи наляво (px)
        - confidence_threshold: Минимална confidence за приемане на детекции
        - sharpening_strength: Коефициент за шарпеникг (1.0 = без шарпеникг)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.debug_folder = os.path.join(output_folder, "debug_images")
        self.auto_split = auto_split
        
        # Configurable parameters
        self.word_spacing_tolerance = word_spacing_tolerance
        self.left_word_penalty_threshold = left_word_penalty_threshold
        self.confidence_threshold = confidence_threshold
        self.sharpening_strength = sharpening_strength
        
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.debug_folder, exist_ok=True)

        if OCR_ENGINE == "easyocr":
            print("Зареждане на EasyOCR...")
            self.reader = easyocr.Reader(["en"], gpu=False)
            print("Модел зареден.")

    # ---------------------------
    # IMAGE PREPROCESSING
    # ---------------------------
    def enhance_contrast(self, gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def deskew(self, image):
        coords = np.column_stack(np.where(image < 255))
        if len(coords) == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return image
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def preprocess_for_tesseract(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self.enhance_contrast(gray)
        resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        binary = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        binary = self.deskew(binary)
        return binary

    def preprocess_for_easyocr(self, image):
        """
        Preprocessing с SHARPENING (Unsharp Mask) и CLAHE
        Подобрена версия с адаптивни параметри и CLAHE
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE първо за адаптивна контрастност (НОВО)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Unsharp Mask: gray - blur = sharpened
        # Адаптивна формула: sharpened = enhanced * strength - blur * (strength - 1)
        gaussian_blur = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
        sharpened = cv2.addWeighted(enhanced, self.sharpening_strength, 
                                   gaussian_blur, -(self.sharpening_strength - 1), 0)
        
        # Дескю поддръжка за EasyOCR (НОВО)
        deskewed = self.deskew(sharpened)
        
        return deskewed

    # ---------------------------
    # PAGE DETECTION - ПОДОБРЕНО
    # ---------------------------
    def split_book_pages(self, image, debug_path=None):
        """Подобрено разделяне БЕЗ загуба на текст"""
        height, width = image.shape[:2]
        
        # Търсим тъмна област в средата
        search_start = int(width * 0.45)
        search_end = int(width * 0.55)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        central_region = gray[:, search_start:search_end]
        vertical_darkness = np.mean(central_region, axis=0)
        local_split = np.argmin(vertical_darkness)
        split_x = search_start + local_split
        
        # ВАЖНО: БЕЗ PADDING! Не отрязваме текст
        left = image[:, :split_x]
        right = image[:, split_x:]
        
        # Debug визуализация
        if debug_path:
            vis = image.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            # Рисуваме split линията
            cv2.line(vis, (split_x, 0), (split_x, height), (0, 0, 255), 3)
            cv2.putText(vis, f"SPLIT at x={{split_x}}", (split_x + 10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(debug_path, vis)
            print(f"   📊 Split визуализация: {{debug_path}}")
            print(f"   ✂️  Split позиция: x={{split_x}} (ляво={{split_x}}px, дясно={{width-split_x}}px)")
        
        return [left, right]

    def detect_pages(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Грешка при зареждане: {{image_path}}")
            return []
        
        h, w = img.shape[:2]
        
        # Проверяваме дали трябва да разделим
        if self.auto_split and w / h > 1.1:
            print(f"📖 Разпозната книга (две страници) - aspect ratio: {{w/h:.2f}}")
            base_name = Path(image_path).stem
            debug_path = os.path.join(self.debug_folder, f"{{base_name}}_split.jpg")
            return self.split_book_pages(img, debug_path)
        else:
            if self.auto_split:
                print(f"📄 Една страница - aspect ratio: {{w/h:.2f}}")
            else:
                print(f"📄 Auto-split изключен - третираме като една страница")
            return [img]

    # ---------------------------
    # TEXT POST PROCESSING
    # ---------------------------
    def post_process_text(self, text):
        if not text:
            return text
        text = re.sub(r"(?<!^)[ ]{{2,}}", " ", text, flags=re.MULTILINE)
        text = re.sub(r"\. ([a-z])", lambda m: ". " + m.group(1).upper(), text)
        return text

    # ---------------------------
    # VISUALIZATION
    # ---------------------------
    def visualize_detections(self, image, results, output_path, conf_threshold=0.0):
        """Визуализира ВСИЧКИ detected boxes"""
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        if not results:
            cv2.imwrite(output_path, vis_image)
            return
        
        # Сортираме всички резултати
        all_results = [(bbox, text, conf) for (bbox, text, conf) in results]
        all_results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        
        # Намираме левия margin
        doc_left_margin = min([bbox[0][0] for bbox, text, conf in all_results])
        
        # Вертикална линия за левия margin
        cv2.line(vis_image, (int(doc_left_margin), 0), (int(doc_left_margin), vis_image.shape[0]), 
                 (255, 0, 255), 2)
        
        # Рисуваме ВСИЧКИ думи
        for bbox, text, conf in all_results:
            # Цвят според confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Зелен
            elif conf > 0.6:
                color = (0, 255, 255)  # Жълт
            elif conf > 0.4:
                color = (0, 165, 255)  # Оранжев
            elif conf > 0.2:
                color = (100, 100, 255)  # Светло червен
            else:
                color = (0, 0, 255)  # Червен (много нисък)
            
            # Bounding box
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis_image, [points], True, color, 2)
            
            # Информация
            top_left = (int(bbox[0][0]), int(bbox[0][1]) - 5)
            info_text = f"c:{{conf:.2f}}"
            cv2.putText(vis_image, info_text, top_left, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        cv2.imwrite(output_path, vis_image)
        print(f"   📊 Debug визуализация: {{output_path}}")
        print(f"   📈 Статистика: {{len(results)}} думи визуализирани")

    # ---------------------------
    # OCR ENGINES
    # ---------------------------
    def extract_text_easyocr(self, image, debug_name=None):
        processed = self.preprocess_for_easyocr(image)
        results = self.reader.readtext(processed, detail=1, paragraph=False)
        
        if not results:
            return ""
        
        # Визуализация
        if debug_name:
            debug_path = os.path.join(self.debug_folder, f"{{debug_name}}_detections.jpg")
            self.visualize_detections(processed, results, debug_path, conf_threshold=0.0)
        
        # Принтираме detected текст
        print(f"\n   🔍 Detected текст (първите 20):")
        for i, (bbox, text, conf) in enumerate(results[:20]):
            x, y = int(bbox[0][0]), int(bbox[0][1])
            print(f"      {{i+1}}. [{{x:4d}},{{y:4d}}] conf={{conf:.2f}} '{{text}}'")
        
        # ПОДОБРЕНО: Мягко филтриране по confidence (ПРОМЕНЕНО)
        print(f"\n   ⚠️  Confidence filtering: conf >= {{self.confidence_threshold}}")
        filtered = [r for r in results if r[2] > self.confidence_threshold]
        
        if not filtered:
            print(f"   ⚠️  След филтриране (conf > {{self.confidence_threshold}}): 0 думи!")
            print(f"   💡 Hint: Понижи confidence_threshold ако имаш нулеви резултати")
            return ""
        
        print(f"   ✅ Приети: {{len(filtered)}}/{{len(results)}} думи")

        # Намираме левия margin за табулация
        doc_left_margin = min([bbox[0][0] for bbox, text, conf in filtered])
        print(f"   📏 Левия margin: {{int(doc_left_margin)}}px")
        
        # НОВА ЛОГИКА: Човешки-подобно групиране на думи в редове
        print(f"   🧠 Използване на човешки-подобен flow-based алгоритъм за line grouping")
        
        lines = self.group_words_into_lines_human_like(filtered, image.shape[0], doc_left_margin)
        
        final_text = "\n".join(lines)
        return self.post_process_text(final_text)
    
    def group_words_into_lines_human_like(self, filtered_words, image_height, doc_left_margin):
        """
        Групира думи в редове следвайки flow-а на текста като човек
        Оптимизирано със конфигурируеми параметри
        """
        if not filtered_words:
            return []
        
        # Подготвяме думите с позиционна информация
        words_with_pos = []
        for bbox, text, conf in filtered_words:
            x = bbox[0][0]  # left
            y = bbox[0][1]  # top
            width = bbox[1][0] - bbox[0][0]
            height = bbox[2][1] - bbox[0][1]
            words_with_pos.append({
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'text': text,
                'used': False
            })
        
        lines = []
        line_num = 0
        base_y_tolerance = image_height * 0.015
        
        while True:
            # Намираме първата неизползвана дума (най-горе, depois най-ляво)
            unused = [w for w in words_with_pos if not w['used']]
            if not unused:
                break
            
            unused.sort(key=lambda w: (w['y'], w['x']))
            start_word = unused[0]
            start_word['used'] = True
            
            # Започваме нов ред
            line_num += 1
            current_line_words = [start_word]
            current_x = start_word['x'] + start_word['width']
            current_y = start_word['y']
            current_height = start_word['height']
            
            y_tolerance = max(base_y_tolerance, current_height * 0.7)
            
            # Следваме текста надясно
            max_iterations = len(unused)
            for iteration in range(max_iterations):
                # Кандидати: неизползвани думи които са надясно
                # ПОДОБРЕНО: Използване на конфигурируемо word_spacing_tolerance
                candidates = [
                    w for w in words_with_pos 
                    if not w['used'] and w['x'] >= current_x - self.word_spacing_tolerance
                ]
                
                if not candidates:
                    break
                
                # Скориране базирано на X и Y близост
                def score_candidate(w):
                    x_dist = abs(w['x'] - current_x)
                    y_dist = abs(w['y'] - current_y)
                    
                    # Penalty за думи наляво (ПОДОБРЕНО: конфигурируемо)
                    if w['x'] < current_x - self.left_word_penalty_threshold:
                        x_dist += 2000
                    
                    return x_dist * 0.4 + y_dist * 0.6
                
                candidates.sort(key=score_candidate)
                next_word = candidates[0]
                
                # Проверки за нов ред
                y_diff = abs(next_word['y'] - current_y)
                x_jump = next_word['x'] - current_x
                
                # Адаптивна толеранция
                adaptive_tolerance = y_tolerance * (1 + len(current_line_words) * 0.2)
                
                is_new_line = (
                    y_diff > adaptive_tolerance or
                    x_jump < -200
                )
                
                if is_new_line:
                    break
                
                # Добавяме към реда
                next_word['used'] = True
                current_line_words.append(next_word)
                current_x = next_word['x'] + next_word['width']
                current_y = current_y * 0.65 + next_word['y'] * 0.35
                current_height = max(current_height, next_word['height'])
                y_tolerance = max(base_y_tolerance, current_height * 0.7)
            
            # Финализираме реда
            current_line_words.sort(key=lambda w: w['x'])
            first_x = current_line_words[0]['x']
            indent_pixels = first_x - doc_left_margin
            
            # Табулация
            if indent_pixels > 5:
                spaces_count = int(indent_pixels / 5)
                indent = " " * spaces_count
            else:
                indent = ""
            
            line_text = " ".join([w['text'] for w in current_line_words])
            
            # Auto-bullet detection
            if indent_pixels > 25 and line_text:
                first_word = line_text.split()[0] if line_text.split() else ""
                if first_word and first_word[0].islower() and first_word not in ["and", "or", "to", "in", "of", "for", "a", "the"]:
                    line_text = "• " + line_text
            
            print(f"   LINE {{line_num}}: indent={{int(indent_pixels)}}px, {{len(current_line_words)}} words → '{{line_text[:70]}}...'")
            
            lines.append(indent + line_text)
        
        return lines

    def extract_text_tesseract(self, image):
        processed = self.preprocess_for_tesseract(image)
        pil_image = Image.fromarray(processed)
        config = r"--oem 3 --psm 3 -l eng"
        text = pytesseract.image_to_string(pil_image, config=config)
        return self.post_process_text(text)

    def extract_text(self, image, debug_name=None):
        if OCR_ENGINE == "easyocr":
            return self.extract_text_easyocr(image, debug_name)
        elif OCR_ENGINE == "tesseract":
            return self.extract_text_tesseract(image)
        return ""

    # ---------------------------
    # MAIN PROCESSING
    # ---------------------------
    def process_image(self, image_path):
        print(f"\n{{'='*70}}")
        print(f"📄 Обработка: {{image_path}}")
        print(f"{{'='*70}}")
        
        pages = self.detect_pages(image_path)
        print(f"📑 Открити страници: {{len(pages)}}")
        base_name = Path(image_path).stem

        for i, page in enumerate(pages, 1):
            print(f"\n--- Страница {{i}}/{{len(pages)}} ---")
            
            debug_name = f"{{base_name}}_page{{i}}" if len(pages) > 1 else base_name
            text = self.extract_text(page, debug_name)
            
            if not text.strip():
                print(f"⚠️  Няма разпознат текст на страница {{i}}.")
                continue

            if len(pages) > 1:
                name = "left" if i == 1 else "right"
                output_file = os.path.join(self.output_folder, f"{{base_name}}_{{name}}.txt")
            else:
                output_file = os.path.join(self.output_folder, f"{{base_name}}.txt")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\n✅ Записан: {{output_file}}")

    def process_all_images(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = [
            f for f in os.listdir(self.input_folder)
            if Path(f).suffix.lower() in image_extensions
        ]

        if not image_files:
            print("Няма намерени изображения в текущата папка.")
            return

        print(f"🔎 Намерени {{len(image_files)}} файла.\n")

        for image_file in image_files:
            try:
                full_path = os.path.join(self.input_folder, image_file)
                self.process_image(full_path)
            except FileNotFoundError as e:
                print(f"❌ Файл не намерен: {{image_file}} - {{e}}")
            except ValueError as e:
                print(f"❌ Невалидно изображение: {{image_file}} - {{e}}")
            except Exception as e:
                print(f"❌ Неочаквана грешка при {{image_file}}: {{type(e).__name__}}: {{e}}")

        print(f"\n{{'='*70}}")
        print("✅ Готово! Резултатите са в папка 'output'.")
        print(f"📊 Debug изображения са в папка 'output/debug_images'.")
        print(f"{{'='*70}}")


def main():
    print("=" * 70)
    print("OCR СКРИПТ С ПОДОБРЕНО РАЗДЕЛЯНЕ НА СТРАНИЦИ")
    print("=" * 70)
    
    # ВАЖНО: Промени параметрите при нужда
    # confidence_threshold=0.3 филтрира слаби детекции
    # sharpening_strength=1.3 контролира шарпеникг интензивност
    # word_spacing_tolerance=80 контролира разстоянието между думи
    ocr = DocumentOCR(
        input_folder=".",
        output_folder="output",
        auto_split=False,
        confidence_threshold=0.3,  # Минимална confidence
        sharpening_strength=1.3,   # Шарпеникг интензивност
        word_spacing_tolerance=80  # X-разстояние толеранция
    )
    ocr.process_all_images()


if __name__ == "__main__":
    main()
