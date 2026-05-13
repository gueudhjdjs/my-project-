from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle, pandas as pd, sqlite3, hashlib, os, numpy as np, cv2, re
from datetime import datetime
from PIL import Image
import pytesseract
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, email TEXT UNIQUE, password TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, news_text TEXT, prediction TEXT, confidence REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

init_db()

hp = lambda p: hashlib.sha256(p.encode()).hexdigest()
logged = lambda: 'user_id' in session
allowed = lambda f: '.' in f and f.rsplit('.', 1)[1].lower() in {'png','jpg','jpeg','gif','bmp'}

def clean(t):
    t = ' '.join(t.split())
    t = re.sub(r'[^\w\s.,!?;:\'-]', ' ', t)
    return ' '.join([w for w in t.split() if len(w)>1 or w.lower() in ['a','i']]).strip()

def aug(t):
    return f"WASHINGTON (Reuters) - {t} The information has been reported by news organizations." if len(t.split())<40 else t

def ocr(p):
    try:
        img = cv2.imread(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d = cv2.fastNlMeansDenoising(g)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        c = cl.apply(d)
        _, th = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(Image.fromarray(th), config=r'--oem 3 --psm 6')
        return clean(txt) or "No text"
    except: return "Error"

def conf(v):
    try:
        return max(model.predict_proba(v)[0])*100 if hasattr(model,'predict_proba') else (1/(1+np.exp(-model.decision_function(v)[0])))*100
    except: return min(abs(model.decision_function(v)[0])*10, 100)

def save(uid, txt, pred, c):
    conn = sqlite3.connect('users.db')
    conn.execute('INSERT INTO predictions VALUES (NULL,?,?,?,?,CURRENT_TIMESTAMP)', (uid,txt,pred,c))
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return redirect(url_for('detect')) if logged() else render_template('home.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u, p = request.form.get('username'), request.form.get('password')
        if u and p:
            conn = sqlite3.connect('users.db')
            user = conn.execute('SELECT id,username,password FROM users WHERE username=?', (u,)).fetchone()
            conn.close()
            if user and user[2] == hp(p):
                session['user_id'], session['username'] = user[0], user[1]
                flash('Login successful!', 'success')
                return redirect(url_for('detect'))
            flash('Invalid credentials!', 'error')
        else: flash('Fill all fields!', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        u, e, p, cp = request.form.get('username'), request.form.get('email'), request.form.get('password'), request.form.get('confirm_password')
        if u and e and p and cp:
            if p != cp: flash('Passwords do not match!', 'error')
            elif len(p) < 6: flash('Password too short!', 'error')
            else:
                try:
                    conn = sqlite3.connect('users.db')
                    conn.execute('INSERT INTO users VALUES (NULL,?,?,?,CURRENT_TIMESTAMP)', (u,e,hp(p)))
                    conn.commit()
                    conn.close()
                    flash('Account created!', 'success')
                    return redirect(url_for('login'))
                except: flash('Username/email exists!', 'error')
        else: flash('Fill all fields!', 'error')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out!', 'success')
    return redirect(url_for('home'))

@app.route('/detect', methods=['GET','POST'])
def detect():
    if not logged(): return redirect(url_for('login'))
    pt = ct = cft = ni = fm = None
    if request.method == 'POST':
        ni = request.form.get('news')
        if ni:
            v = vectorizer.transform([ni])
            pred = model.predict(v)[0]
            c = conf(v)
            pt, ct, cft = f"The news is: {pred}", "Category: General", f"Confidence: {round(c,2)}%"
            save(session['user_id'], ni, pred, c)
    return render_template('index.html', prediction_text=pt, category_text=ct, confidence_text=cft, news_input=ni, feedback_msg=fm, username=session.get('username'))

@app.route('/image_detect', methods=['GET','POST'])
def image_detect():
    if not logged(): return redirect(url_for('login'))
    pt = ct = et = er = None
    if request.method == 'POST':
        if 'image' not in request.files: er = 'No file'
        else:
            f = request.files['image']
            if f.filename == '': er = 'No file selected'
            elif f and allowed(f.filename):
                fn = secure_filename(f.filename)
                fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
                f.save(fp)
                et = ocr(fp)
                if et and not et.startswith('Error') and et != "No text":
                    wc = len(et.split())
                    if wc >= 5:
                        a = aug(et)
                        v = vectorizer.transform([a])
                        pred = model.predict(v)[0]
                        c = conf(v)
                        pt = f"The news is: {pred}"
                        ct = f"Confidence: {round(c,2)}% ({wc} words)"
                        save(session['user_id'], f"[IMG]{et[:200]}", pred, c)
                    else: er = f'Only {wc} words extracted'
                else: er = 'Could not extract text'
                if os.path.exists(fp): os.remove(fp)
            else: er = 'Invalid file type'
    return render_template('image_detect.html', prediction_text=pt, confidence_text=ct, extracted_text=et, error=er, username=session.get('username'))

@app.route('/bulk', methods=['GET','POST'])
def bulk_news():
    if not logged(): return redirect(url_for('login'))
    res = er = None
    if request.method == 'POST':
        try:
            f = request.files.get('file')
            if not f: er = "Upload CSV"
            else:
                df = pd.read_csv(f)
                col = next((c for c in df.columns if c.strip().lower() in ['news','text','article','content','headline']), None)
                if not col: er = "No valid column"
                else:
                    nl = df[col].dropna().astype(str).tolist()
                    preds = model.predict(vectorizer.transform(nl))
                    res = list(zip(nl, preds))
        except Exception as e: er = f"Error: {e}"
    return render_template('bulk.html', result=res, error=er, username=session.get('username'))

@app.route('/history')
def history():
    if not logged(): return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    preds = conn.execute('SELECT news_text,prediction,confidence,created_at FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 50', (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html', predictions=preds, username=session.get('username'))

@app.route('/dashboard')
def dashboard():
    return redirect(url_for('bulk_news')) if logged() else redirect(url_for('login'))

@app.route('/feedback', methods=['POST'])
def feedback():
    if not logged(): return redirect(url_for('login'))
    n, uf = request.form.get('news'), request.form.get('feedback')
    fm = "✅ Thanks!" if n and uf and open('feedback.txt','a').write(f"User:{session.get('username')}\nNews:{n}\nFeedback:{uf}\nDate:{datetime.now()}\n\n") or True else "⚠ Failed"
    return render_template('index.html', feedback_msg=fm, username=session.get('username'))

if __name__ == '__main__':
    app.run(debug=True)
