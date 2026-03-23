# 🐍 Python Virtual Environment (venv) Guide

This project uses a Python virtual environment (`venv`) to manage dependencies locally. All required packages are listed in the `requirements.txt` file located in the project root.

---

## 📦 1. Create the Virtual Environment

Run this in the project root directory:

```bash
python3 -m venv venv
```

This will create a `venv/` folder containing an isolated Python environment.

---

## ▶️ 2. Activate the Virtual Environment

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

When activated, your terminal should show:

```
(venv) ...
```

---

## 📥 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ➕ 4. Installing New Packages

If you install additional packages:

```bash
pip install package_name
```

Update the requirements file:

```bash
pip freeze > requirements.txt
```

---

## ▶️ 5. Running Python Scripts

Make sure the virtual environment is activated, then run:

```bash
python your_script.py
```

---

## ❌ 6. Deactivate the Virtual Environment

When you're done working:

```bash
deactivate
```

---

## 🧹 7. Recreating the Environment (Clean Setup)

If needed, delete and recreate the environment:

```bash
rm -rf venv        # macOS/Linux
rmdir /s venv      # Windows

python3 -m venv venv
pip install -r requirements.txt
```

---

## ⚙️ 8. VS Code Setup (Recommended)

1. Open Command Palette:

   * `Ctrl + Shift + P` (Windows)
   * `Cmd + Shift + P` (Mac)

2. Select:

   ```
   Python: Select Interpreter
   ```

3. Choose:

   ```
   ./venv/bin/python
   ```

---

## 📁 Notes

* The `venv/` folder should NOT be committed to version control.
* Ensure `.gitignore` includes:

```
venv/
__pycache__/
```

---

## ✅ Summary

* Create → `python3 -m venv venv`
* Activate → `source venv/bin/activate`
* Install → `pip install -r requirements.txt`
* Work → `python script.py`
* Deactivate → `deactivate`

---

This setup ensures consistent environments across different machines and avoids dependency conflicts.
