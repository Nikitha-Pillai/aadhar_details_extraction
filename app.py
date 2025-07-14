from flask import Flask, render_template, request
from paddleocr import PaddleOCR
import re
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False)

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def count_horizontal_lines(results, threshold=1.5):
    """Count lines where width > threshold * height (i.e., horizontal lines)."""
    if not results or not results[0]:
        return 0
    count = 0
    for line in results[0]:
        box = line[0]
        width = np.linalg.norm(np.array(box[0]) - np.array(box[1]))
        height = np.linalg.norm(np.array(box[0]) - np.array(box[3]))
        if width > threshold * height:
            count += 1
    return count

def rotate_image(image, angle):
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def auto_correct_image_orientation(image_path, save_corrected=True):
    image = cv2.imread(image_path)
    best_angle = 0
    best_rotated = image
    max_horizontal = -1

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(image, angle)
        temp_path = "temp_rotated.jpg"
        cv2.imwrite(temp_path, rotated)
        result = ocr.ocr(temp_path, cls=False)
        horizontal_count = count_horizontal_lines(result)
        print(f"Angle {angle}: {horizontal_count} horizontal lines")
        if horizontal_count > max_horizontal:
            max_horizontal = horizontal_count
            best_angle = angle
            best_rotated = rotated

    if save_corrected:
        output_path = os.path.splitext(image_path)[0] + "_auto_rotated.jpg"
        cv2.imwrite(output_path, best_rotated)
        print(f"✅ Best orientation: {best_angle}°, saved to {output_path}")
        return output_path
    else:
        print(f"✅ Best orientation: {best_angle}° (not saved)")
        return best_rotated

# Check if image needs rotation
def rotate_if_needed(image_path):
    return auto_correct_image_orientation(image_path, save_corrected=True)


    
# Enhance image for better OCR
def enhance_image(image_path):
    img = image_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(enhanced, 2.0, blur, -1.0, 0)
    if sharp.shape[1] < 800:
        sharp = cv2.resize(sharp, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    out_path = os.path.splitext(image_path)[0] + "_enhanced.jpg"
    cv2.imwrite(out_path, sharp)
    return out_path

# Extract address if it's on the left side
def extract_left(ocr_results, image_path):
    image = cv2.imread(image_path)
    lines = sorted(ocr_results[0], key=lambda x: sum([pt[1] for pt in x[0]]) / 4)

    address_started = False
    start_y = None
    address_texts = []
    address_boxes = []

    for box, (text, score) in lines:
        if score < 0.3:
            continue
        text = text.strip()
        center_y = sum([pt[1] for pt in box]) / 4
        left_x = min(pt[0] for pt in box)

        if not address_started and "address" in text.lower():
            address_started = True
            start_y = center_y
            address_texts.append(text)
            address_boxes.append(box)
            continue

        if address_started:
            if center_y <= start_y:
                continue
            if "uidai" in text.lower() or "www" in text.lower() or re.fullmatch(r'\d{4} \d{4} \d{4}', text):
                break
            if left_x < image.shape[1] * 0.25:
                continue
            address_texts.append(text)
            address_boxes.append(box)

    # Deduplicate
    combined = " ".join(address_texts)
    final_texts = []
    final_boxes = []

    for t, b in zip(address_texts, address_boxes):
        if t.isdigit() and len(t) == 6:
            if any(t in other and other != t for other in address_texts):
                continue
        final_texts.append(t)
        final_boxes.append(b)

    if final_texts and final_texts[-1].isdigit() and len(final_texts[-1]) == 6:
        pin = final_texts.pop()
        final_boxes.pop()
        final_texts[-1] += " " + pin

    return final_texts

# Determine address block position
def extract_pos(results, img_path):
    import numpy as np
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    boxes = []
    found_address = False
    for line in results[0]:
        text = line[1][0].lower()
        box = line[0]
        left_x = min(pt[0] for pt in box)

        if not found_address and "address" in text:
            found_address = True
            boxes.append(box)
            continue

        if found_address:
            # Stop if it's clearly outside address block
            if any(stop in text for stop in ['uidai', 'www', 'gov', 'aadhar', 'help']):
                break
            # Only take left-aligned boxes (safe assumption for Indian Aadhaar back)
            if left_x < w * 0.5:
                boxes.append(box)

    if not boxes:
        return "not found"

    all_points = np.concatenate(boxes, axis=0)
    center_x = int(np.mean(all_points[:, 0]))
    center_y = int(np.mean(all_points[:, 1]))

    horizontal = 'left' if center_x < w / 3 else 'middle' if center_x < w * 2 / 3 else 'right'
    vertical = 'top' if center_y < h / 3 else 'middle' if center_y < h * 2 / 3 else 'bottom'

    print(f"Position: {vertical} {horizontal}")
    return vertical + ' ' + horizontal




# Extract first and last name
def extract_name(name):
    if '.' in name:
        parts = [p.strip() for p in name.split('.') if p.strip()]
        if len(parts) >= 3:
            return parts[0], parts[-2] + parts[-1]
        return parts[0], parts[-1] if len(parts) > 1 else ''
    elif ' ' in name:
        parts = name.split()
        if len(parts) >= 3:
            return parts[0], parts[-2] + parts[-1]
        return parts[0], parts[-1] if len(parts) > 1 else ''
    else:
        return name.strip(), ''

# Main function to extract details
def extrct_det(img_path_f, img_path_b, already_enhanced=False):
    resultf = ocr.ocr(img_path_f, cls=True)
    resultb = ocr.ocr(img_path_b, cls=True)
    textf = [line[1][0] for line in resultf[0]]
    textb = [line[1][0] for line in resultb[0]]

    name = dob = adhar_no = address = ""
    in_line = ""

    # Name
    name = next((line.strip() for line in textf if not re.search(r'\d', line) and len(line.strip()) > 5 and not any(k in line.lower() for k in ['government', 'india', 'male', 'female', 'dob'])), None)
    fname, lname = extract_name(name) if name else ("Name not found", "Name not found")

    # DOB
    for line in textf:
        date = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', line)
        if date:
            dob = date.group()
            break

    # Aadhaar Number
    for line in textf:
        cline = line.strip().replace(' ', '')
        if re.fullmatch(r'\d{12}', cline):
            in_line = line
            adhar_no = cline[:4] + ' ' + cline[4:8] + ' ' + cline[8:]
            break

    # Address
    pos = extract_pos(resultb, img_path_b)
    if pos in ['top middle', 'top right', 'middle middle', 'middle right']:
        address_lines = extract_left(resultb, img_path_b)
        address = " ".join(address_lines)
    else:
        address_keywords = ['acdress', 'c/o', 's/o', 'd/o', 'w/o','address']
        stop_keywords = ['vid', 'www', 'uidai', 'gov', 'aadhar', 'aadhaar']
        for i, line in enumerate(textb):
            line_lower = line.lower()
            if any(k in line_lower for k in address_keywords):
                address_lines = []
                for addr_line in textb[i:]:
                    addr_line_lower = addr_line.lower()
                    if any(phrase in addr_line_lower for phrase in ['print date', 'issued on', 'issued', 'date']):
                        continue
                    if any(re.search(r'\b' + stop + r'\b', addr_line_lower) for stop in stop_keywords):
                        break
                    if re.fullmatch(r'\d{4} \d{4} \d{4}', addr_line) or re.match(in_line, addr_line):
                        break
                    if re.search(r'\b\d{6}\b', addr_line):
                        address_lines.append(addr_line)
                        break
                    address_lines.append(addr_line)
                address = " ".join(address_lines)
                address = re.sub(r'(?i)(address|c/o|s/o|d/o|w/o)[: ]*', '', address).strip()
                break

    # Retry logic
    if (not name or not dob or not adhar_no or not address):
        if not already_enhanced:
            print("Details could not be extracted. Trying again with enhanced images...")
            enhanced_f = enhance_image(img_path_f)
            enhanced_b = enhance_image(img_path_b)
            return extrct_det(enhanced_f, enhanced_b, already_enhanced=True)
        else:
            print("Details could not be extracted even after enhancement.")
            return {
                'First Name': fname,
                'Last Name': lname,
                'Date of Birth': dob or "DOB not found",
                'Aadhaar Number': adhar_no or "Aadhaar number not found",
                'Address': address or "Address not found"
            }

    return {
        'First Name': fname,
        'Last Name': lname,
        'Date of Birth': dob,
        'Aadhaar Number': adhar_no,
        'Address': address
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    details = {}
    if request.method == 'POST':
        front_img = request.files['front_img']
        back_img = request.files['back_img']

        if front_img and back_img:
            front_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(front_img.filename))
            back_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(back_img.filename))
            front_img.save(front_path)
            back_img.save(back_path)

            rotated_front = rotate_if_needed(front_path)
            rotated_back = rotate_if_needed(back_path)

            details = extrct_det(rotated_front, rotated_back)

    return render_template('index.html', details=details)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
