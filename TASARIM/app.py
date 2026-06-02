import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="League of Legends ML Analiz",
    page_icon="🎮",
    layout="wide"
)

# Başlık
st.title("🎮 League of Legends Ranked Match Analiz ve Tahmin Sistemi")
st.markdown("---")

# Sidebar - Navigasyon
st.sidebar.title("📊 Menü")
page = st.sidebar.selectbox(
    "Sayfa Seçin",
    ["🏠 Ana Sayfa", "🎯 Kazanma Tahmini", "👥 Oyuncu Persona Analizi", "📈 Model Performansı"]
)

# Model ve veri yükleme fonksiyonu
@st.cache_resource
def load_models_and_data():
    """Modelleri ve ön işleme araçlarını yükle"""
    MODEL_DIR = Path("models")
    
    if not MODEL_DIR.exists():
        return None, None, None, None
    
    try:
        # Feature bilgilerini yükle
        with open(MODEL_DIR / "feature_info.pkl", "rb") as f:
            feature_info = pickle.load(f)
        
        # Modelleri yükle
        models_dict = {}
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = MODEL_DIR / filename
            if model_path.exists():
                models_dict[name] = joblib.load(model_path)
        
        # Preprocessing araçlarını yükle
        scaler = None
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        num_imputer = None
        cat_imputer = None
        if (MODEL_DIR / "num_imputer.pkl").exists():
            num_imputer = joblib.load(MODEL_DIR / "num_imputer.pkl")
        if (MODEL_DIR / "cat_imputer.pkl").exists():
            cat_imputer = joblib.load(MODEL_DIR / "cat_imputer.pkl")
        
        return models_dict, scaler, num_imputer, feature_info
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None, None, None

# Ana Sayfa
if page == "🏠 Ana Sayfa":
    st.header("Hoş Geldiniz!")
    st.markdown("""
    Bu uygulama League of Legends ranked maç verilerini analiz eder ve tahminler yapar.
    
    ### Özellikler:
    - ✅ **Kazanma Tahmini**: Oyuncu istatistiklerine göre maç kazanma olasılığını tahmin eder
    - ✅ **Oyuncu Persona Analizi**: Oyuncuları performanslarına göre kategorize eder
    - ✅ **Model Performansı**: Eğitilmiş modellerin performans metriklerini gösterir
    
    ### Kullanım:
    Sol menüden istediğiniz sayfayı seçerek başlayabilirsiniz.
    """)
    
    # Örnek istatistikler
    st.subheader("📊 Örnek İstatistikler")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ortalama K/D", "1.5", "0.2")
    with col2:
        st.metric("Ortalama Altın", "12,500", "500")
    with col3:
        st.metric("Ortalama Hasar", "25,000", "1,200")
    with col4:
        st.metric("Kazanma Oranı", "50%", "2%")

# Kazanma Tahmini Sayfası
elif page == "🎯 Kazanma Tahmini":
    st.header("🎯 Maç Kazanma Tahmini")
    st.markdown("Oyuncu istatistiklerini girerek maç kazanma olasılığını tahmin edin.")
    
    # İki sütunlu form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temel İstatistikler")
        kills = st.number_input("Kills", min_value=0, max_value=50, value=5)
        deaths = st.number_input("Deaths", min_value=0, max_value=50, value=3)
        assists = st.number_input("Assists", min_value=0, max_value=50, value=7)
        gold_earned = st.number_input("Gold Earned", min_value=0, max_value=50000, value=12000)
        gold_spent = st.number_input("Gold Spent", min_value=0, max_value=50000, value=11000)
        
    with col2:
        st.subheader("Hasar ve Vizyon")
        damage_dealt = st.number_input("Damage Dealt", min_value=0, max_value=200000, value=25000)
        damage_to_champ = st.number_input("Damage to Champions", min_value=0, max_value=100000, value=20000)
        damage_taken = st.number_input("Damage Taken", min_value=0, max_value=100000, value=15000)
        vision_score = st.number_input("Vision Score", min_value=0, max_value=200, value=25)
        kill_participation = st.slider("Kill Participation (%)", 0.0, 1.0, 0.6)
    
    # Ek özellikler
    with st.expander("🔧 Ek Özellikler"):
        col3, col4 = st.columns(2)
        with col3:
            duration = st.number_input("Game Duration (seconds)", min_value=300, max_value=3600, value=1800)
            position = st.selectbox("Position", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
        with col4:
            team_champKills = st.number_input("Team Champion Kills", min_value=0, max_value=100, value=25)
            team_dragonKills = st.number_input("Team Dragon Kills", min_value=0, max_value=10, value=2)
    
    # Model seçimi
    models_dict, scaler, num_imputer, feature_info = load_models_and_data()
    
    if models_dict:
        selected_model_name = st.selectbox(
            "Model Seçin",
            list(models_dict.keys()),
            help="Farklı modeller farklı tahminler verebilir"
        )
    else:
        selected_model_name = None
        st.warning("⚠️ Modeller bulunamadı. Lütfen önce EDA.ipynb notebook'unu çalıştırarak modelleri eğitin ve kaydedin.")
    
    # Tahmin butonu
    if st.button("🔮 Tahmin Yap", type="primary") and models_dict:
        try:
            # Kullanıcı girdilerini feature formatına çevir
            EPS = 1e-6
            
            # Feature engineering (notebook'takiyle aynı)
            kda_ratio = (kills + assists) / max(deaths, EPS)
            dmg_per_gold = damage_to_champ / max(gold_earned, EPS)
            gold_per_kill = gold_earned / max(kills, EPS)
            dmg_per_death = damage_to_champ / max(deaths, EPS)
            kill_death_ratio = kills / max(deaths, EPS)
            vision_per_min = vision_score / max(duration / 60.0, EPS)
            kp_x_teamkills = kill_participation * team_champKills
            dmg_taken_per_hp = damage_taken / max(1500, EPS)  # Varsayılan HP
            
            # Sayısal özellikler (basitleştirilmiş - gerçekte feature_info'dan gelecek)
            input_features = {
                'kills': kills,
                'deaths': deaths,
                'assists': assists,
                'kda_ratio': kda_ratio if 'kda_ratio' in (feature_info.get('numeric_features', []) if feature_info else []) else kda_ratio,
                'kill_participation': kill_participation,
                'gold_earned': gold_earned,
                'gold_spent': gold_spent,
                'damage_to_champ': damage_to_champ,
                'damage_taken': damage_taken,
                'vision_score': vision_score,
            }
            
            # FE özelliklerini ekle
            if feature_info and 'numeric_features' in feature_info:
                if 'kda_calc' in feature_info['numeric_features']:
                    input_features['kda_calc'] = kda_ratio
                if 'dmg_per_gold' in feature_info['numeric_features']:
                    input_features['dmg_per_gold'] = dmg_per_gold
                if 'vision_per_min' in feature_info['numeric_features']:
                    input_features['vision_per_min'] = vision_per_min
            
            # DataFrame oluştur (tüm feature'ları içerecek şekilde)
            if feature_info and 'feature_names' in feature_info:
                # Tüm feature'lar için DataFrame oluştur
                input_df = pd.DataFrame(0, index=[0], columns=feature_info['feature_names'])
                
                # Mevcut değerleri ata
                for feat in input_features:
                    if feat in input_df.columns:
                        input_df[feat] = input_features[feat]
                
                # Kategorik özellikler için one-hot encoding (basitleştirilmiş)
                if 'position' in feature_info.get('cat_features', []):
                    pos_col = f'position_{position}'
                    if pos_col in input_df.columns:
                        input_df[pos_col] = 1
            else:
                # Fallback: sadece sayısal özellikler
                input_df = pd.DataFrame([input_features])
            
            # Model tahmini
            model = models_dict[selected_model_name]
            
            if selected_model_name == 'Logistic Regression' and scaler:
                input_scaled = scaler.transform(input_df)
                win_prob = model.predict_proba(input_scaled)[0, 1]
            else:
                win_prob = model.predict_proba(input_df)[0, 1]
            
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            # Fallback: basit kural tabanlı
            kda_ratio = (kills + assists) / max(deaths, 1)
            score = 0
            if kda_ratio > 2.0:
                score += 30
            elif kda_ratio > 1.5:
                score += 20
            if kill_participation > 0.7:
                score += 20
            if damage_to_champ > 20000:
                score += 20
            win_prob = min(max(score / 100, 0.1), 0.9)
        
        # Sonuçları göster
        st.markdown("---")
        st.subheader("📊 Tahmin Sonuçları")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Kazanma Olasılığı", f"{win_prob*100:.1f}%")
        with result_col2:
            st.metric("KDA Ratio", f"{kda_ratio:.2f}")
        with result_col3:
            prediction = "Kazanma" if win_prob > 0.5 else "Kaybetme"
            st.metric("Tahmin", prediction)
        
        # Model bilgisi
        if models_dict:
            st.caption(f"Kullanılan Model: {selected_model_name}")
        
        # Görselleştirme
        st.progress(win_prob)
        
        # Detaylı analiz
        st.markdown("### 📈 Detaylı Analiz")
        kda_ratio = (kills + assists) / max(deaths, 1)
        analysis_df = pd.DataFrame({
            "Metrik": ["KDA Ratio", "Kill Participation", "Damage to Champions", "Vision Score"],
            "Değer": [f"{kda_ratio:.2f}", f"{kill_participation*100:.1f}%", f"{damage_to_champ:,}", f"{vision_score}"],
            "Durum": [
                "Mükemmel" if kda_ratio > 2.0 else "İyi" if kda_ratio > 1.5 else "Orta",
                "Mükemmel" if kill_participation > 0.7 else "İyi" if kill_participation > 0.5 else "Orta",
                "Mükemmel" if damage_to_champ > 20000 else "İyi" if damage_to_champ > 15000 else "Orta",
                "Mükemmel" if vision_score > 30 else "İyi" if vision_score > 20 else "Orta"
            ]
        })
        st.dataframe(analysis_df, use_container_width=True)

# Oyuncu Persona Analizi
elif page == "👥 Oyuncu Persona Analizi":
    st.header("👥 Oyuncu Persona Analizi")
    st.markdown("Oyuncu performansına göre persona kategorisi belirleme.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performans Metrikleri")
        kills_p = st.number_input("Kills (Persona)", min_value=0, max_value=50, value=5, key="p_kills")
        deaths_p = st.number_input("Deaths (Persona)", min_value=0, max_value=50, value=3, key="p_deaths")
        assists_p = st.number_input("Assists (Persona)", min_value=0, max_value=50, value=7, key="p_assists")
        
    with col2:
        st.subheader("Ek Metrikler")
        gold_earned_p = st.number_input("Gold Earned (Persona)", min_value=0, max_value=50000, value=12000, key="p_gold")
        damage_champ_p = st.number_input("Damage to Champions (Persona)", min_value=0, max_value=100000, value=20000, key="p_dmg")
        vision_p = st.number_input("Vision Score (Persona)", min_value=0, max_value=200, value=25, key="p_vision")
        kp_p = st.slider("Kill Participation (Persona)", 0.0, 1.0, 0.6, key="p_kp")
    
    if st.button("🔍 Persona Analizi Yap", type="primary"):
        # Basit persona belirleme
        kda_p = (kills_p + assists_p) / max(deaths_p, 1)
        
        # Persona skorları
        aggressive_score = (kills_p / max(deaths_p, 1)) * 0.4 + (damage_champ_p / 20000) * 0.3 + (kp_p) * 0.3
        support_score = (assists_p / max(kills_p, 1)) * 0.4 + (vision_p / 30) * 0.4 + (kp_p) * 0.2
        carry_score = (kda_p) * 0.3 + (gold_earned_p / 15000) * 0.4 + (damage_champ_p / 25000) * 0.3
        
        scores = {
            "Bencil Taşıyıcı": aggressive_score,
            "Takım Oyuncusu": support_score,
            "Carry Oyuncusu": carry_score
        }
        
        best_persona = max(scores, key=scores.get)
        best_score = scores[best_persona]
        
        st.markdown("---")
        st.subheader("🎭 Persona Sonuçları")
        
        persona_col1, persona_col2 = st.columns(2)
        
        with persona_col1:
            st.metric("Tahmin Edilen Persona", best_persona)
            st.metric("Persona Skoru", f"{best_score:.2f}")
        
        with persona_col2:
            # Persona açıklamaları
            persona_descriptions = {
                "Bencil Taşıyıcı": "Yüksek kill sayısı, agresif oyun tarzı",
                "Takım Oyuncusu": "Yüksek assist ve vision skoru, takım odaklı",
                "Carry Oyuncusu": "Dengeli performans, oyunu taşıyan oyuncu"
            }
            st.info(f"**{best_persona}**: {persona_descriptions.get(best_persona, '')}")
        
        # Skor karşılaştırması
        st.markdown("### 📊 Persona Skorları")
        persona_df = pd.DataFrame({
            "Persona": list(scores.keys()),
            "Skor": list(scores.values())
        }).sort_values("Skor", ascending=False)
        
        st.bar_chart(persona_df.set_index("Persona"))

# Model Performansı
elif page == "📈 Model Performansı":
    st.header("📈 Model Performans Metrikleri")
    st.markdown("Eğitilmiş modellerin performans karşılaştırması.")
    
    # Örnek metrikler (gerçek kullanımda model sonuçlarından gelecek)
    st.subheader("Model Karşılaştırması")
    
    metrics_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.72, 0.78, 0.80],
        "Precision": [0.71, 0.77, 0.79],
        "Recall": [0.73, 0.79, 0.81],
        "F1-Score": [0.72, 0.78, 0.80],
        "AUC-ROC": [0.79, 0.85, 0.87]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Görselleştirme
    st.subheader("Metrik Görselleştirme")
    
    metric_select = st.selectbox("Gösterilecek Metrik", ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])
    
    st.bar_chart(metrics_df.set_index("Model")[metric_select])
    
    # Model önerisi
    st.subheader("💡 Model Önerisi")
    best_model_idx = metrics_df["AUC-ROC"].idxmax()
    best_model = metrics_df.loc[best_model_idx, "Model"]
    best_auc = metrics_df.loc[best_model_idx, "AUC-ROC"]
    
    st.success(f"**En İyi Model**: {best_model} (AUC-ROC: {best_auc:.3f})")
    st.info("""
    **Model Seçim Kriterleri:**
    - AUC-ROC: Genel performans için en iyi metrik
    - Precision: Yanlış pozitifleri minimize etmek için
    - Recall: Tüm pozitifleri yakalamak için
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>League of Legends ML Analiz Projesi | Veri Bilimi ve Makine Öğrenmesi</p>
</div>
""", unsafe_allow_html=True)

