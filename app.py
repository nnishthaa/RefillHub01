import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")
df=pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar
with st.sidebar:
    st.image("refillhub_logo.png", use_column_width=True)
    st.markdown("## 🌱 What do you want to see?")
    page=st.radio("",["🏠 Dashboard Overview","🧩 About ReFill Hub","📊 Analysis"])
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.write("👑 Nishtha – Insights Lead")
    st.write("✨ Anjali – Data Analyst")
    st.write("🌱 Amatulla – Sustainability Research")
    st.write("📊 Amulya – Analytics Engineer")
    st.write("🧠 Anjan – Strategy & AI")


# Dashboard Overview
if page=="🏠 Dashboard Overview":
    st.markdown("<h1 style='background:linear-gradient(90deg,#6a11cb,#2575fc); padding:20px; border-radius:12px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.write("Smart analytics engine for refill adoption.")

    # ✅ GREEN BOX
    st.markdown("""
        <div style="background-color:#d8f5d0; padding:20px; border-radius:12px; margin-top:20px;">
            <h3 style="color:black;">💡 ReFill Hub: Business Overview</h3>
            <p style="color:black; font-size:16px;">
                The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE 
                for daily essentials such as shampoos, detergents, and cooking oils.<br><br>
                The core mission is to minimize single-use plastic waste by encouraging refillable consumption supported by digital 
                payments and a frictionless user experience.<br><br>
                The business targets young professionals and eco-conscious urban residents.<br>
                Survey insights confirm middle-income groups and sustainability-aware respondents as early adopters.<br><br>
                Future vision includes expanding into non-liquid categories and scaling across the GCC.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("High Eco-Intent", "31.4%")
    c4.metric("Warm Adopters", "48.7%")


# About Page
elif page=="🧩 About ReFill Hub":
    st.title("About ReFill Hub")
    c1,c2=st.columns([1.2,1])
    with c1:
        st.markdown("### 💡 Business Overview")
        st.write("ReFill Hub eliminates single-use plastics using smart automated refill kiosks across UAE.")
        st.markdown("### 🚀 Mission")
        st.write("Making refill culture mainstream across the region.")
        st.markdown("### 🎯 Who We Serve")
        st.write("""
- Young professionals  
- Families  
- Eco-conscious consumers  
- Urban residents across UAE  
""")
    with c2:
        st.markdown("### 💳 Business Model")
        st.write("""
- Refill margins  
        """)
        st.write("""
- Brand partnerships  
- Smart container sales  
- Subscription plans  
""")
        st.markdown("### 🔮 Future Roadmap")
        st.write("UAE rollout → GCC expansion → ReFill OS")


# Analysis
elif page=="📊 Analysis":
    tabs=st.tabs(["Classification","Regression","Clustering","Association Rules","Insights"])

    # Classification
    with tabs[0]:
        st.header("Classification Models")
        df_c=df.copy()
        le=LabelEncoder()
        for col in df_c.select_dtypes(include=['object']).columns:
            df_c[col]=le.fit_transform(df_c[col])
        target="Likely_to_Use_ReFillHub"
        X=df_c.drop(columns=[target])
        y=df_c[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        models={"Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Gradient Boosting":GradientBoostingClassifier()}

        metrics=[]
        cols=st.columns(2)
        idx=0
        for name,m in models.items():
            m.fit(X_train,y_train)
            preds=m.predict(X_test)
            probs=m.predict_proba(X_test)[:,1]

            fig,ax=plt.subplots(figsize=(4,3))
            sns.heatmap(confusion_matrix(y_test,preds),annot=True,fmt="d",cmap="Greens",ax=ax)
            ax.set_title(f"{name} – Confusion Matrix")
            cols[idx%2].pyplot(fig); idx+=1

            fig,ax=plt.subplots(figsize=(4,3))
            fpr,tpr,_=roc_curve(y_test,probs)
            ax.plot(fpr,tpr)
            ax.set_title(f"{name} – ROC Curve")
            cols[idx%2].pyplot(fig); idx+=1

            rep=classification_report(y_test,preds,output_dict=True)
            metrics.append([name,rep['weighted avg']['precision'],rep['weighted avg']['recall'],rep['weighted avg']['f1-score'],accuracy_score(y_test,preds)])

        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(metrics,columns=["Model","Precision","Recall","F1 Score","Accuracy"]))

    # Regression
    with tabs[1]:
        st.header("Willingness to Pay – Regression")
        df_r=df.dropna(subset=["Willingness_to_Pay_AED"])
        df_r2=df_r.copy()
        for col in df_r2.select_dtypes(include=['object']).columns:
            df_r2[col]=le.fit_transform(df_r2[col])
        X=df_r2.drop(columns=["Willingness_to_Pay_AED"])
        y=df_r2["Willingness_to_Pay_AED"]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        reg=LinearRegression().fit(X_train,y_train)
        preds=reg.predict(X_test)
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))

    # Clustering
    with tabs[2]:
        st.header("Customer Clustering")
        k=st.slider("Number of clusters",2,6,3)
        if st.button("Run Clustering"):
            df_num=df.select_dtypes(include=['int64','float64'])
            km=KMeans(n_clusters=k,random_state=42).fit(df_num)
            df['Cluster']=km.labels_
            st.dataframe(df['Cluster'].value_counts())

            p=PCA(n_components=2).fit_transform(df_num)
            fig,ax=plt.subplots()
            sc=ax.scatter(p[:,0],p[:,1],c=df['Cluster'],cmap='viridis')
            plt.colorbar(sc)
            st.pyplot(fig)

    # Association Rules
    with tabs[3]:
        st.header("Association Rules")
        df_ar=df.copy()
        cat=df_ar.select_dtypes(include=['object']).columns
        for col in cat: df_ar[col]=df_ar[col].astype(str)
        df_hot=pd.get_dummies(df_ar[cat]).fillna(0)
        freq=apriori(df_hot,min_support=0.05,use_colnames=True)
        rules=association_rules(freq,metric="lift",min_threshold=1)
        rules=rules[["antecedents","consequents","support","confidence","lift"]].sort_values("lift",ascending=False).head(10)
        st.dataframe(rules)

    # Insights (UPDATED ONLY THIS PART)
    with tabs[4]:
        st.header("Insights")

        # 1️⃣ Eco-aware users show higher adoption
        st.subheader("1. Eco-aware users show higher adoption")
        st.write(
            "Users who regularly choose eco-friendly products demonstrate a much higher likelihood of adopting ReFill Hub. "
            "Their sustainability mindset directly influences refill behavior, making them an ideal early-adopter segment."
        )
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x=df["Uses_Eco_Products"], y=df["Likely_to_Use_ReFillHub"], estimator=np.mean, ax=ax)
        ax.set_xlabel("Uses Eco Products (0=No, 1=Yes)")
        ax.set_title("Eco-Friendly Users vs Adoption Likelihood")
        st.pyplot(fig)

        # 2️⃣ Mid-income groups adopt refills the most
        st.subheader("2. Mid-income consumers show the strongest adoption")
        st.write(
            "Middle-income consumers balance affordability with awareness, and they emerge as the most responsive segment. "
            "They show higher refill likelihood compared to lower or higher-income groups."
        )
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(x=df["Income"], y=df["Likely_to_Use_ReFillHub"], ax=ax)
        ax.set_title("Income Group vs ReFill Adoption Likelihood")
        st.pyplot(fig)

        # 3️⃣ Plastic ban awareness increases interest
        st.subheader("3. Plastic ban awareness strongly boosts interest")
        st.write(
            "Respondents aware of the UAE’s plastic ban are significantly more inclined to try ReFill Hub. "
            "Government regulations act as a powerful motivator for eco-friendly alternatives."
        )
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x=df["Aware_Plastic_Ban"], y=df["Likely_to_Use_ReFillHub"], estimator=np.mean, ax=ax)
        ax.set_xlabel("Awareness of Plastic Ban (0=No, 1=Yes)")
        ax.set_title("Plastic Ban Awareness vs ReFill Adoption")
        st.pyplot(fig)

        # 4️⃣ Sustainability score drives willingness-to-pay
        st.subheader("4. Higher sustainability scores → Higher willingness-to-pay")
        st.write(
            "Users with high reduce-waste scores are willing to pay more for sustainable refill solutions. "
            "Environmental values strongly influence willingness to pay for eco-friendly products."
        )
        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(x=df["Reduce_Waste_Score"], y=df["Willingness_to_Pay_AED"], ax=ax, alpha=0.6)
        ax.set_title("Sustainability Score vs Willingness to Pay")
        st.pyplot(fig)

        # 5️⃣ Location preferences guide deployment
        st.subheader("5. Location preferences guide ideal kiosk placement")
        st.write(
            "Survey data highlights high demand for refill kiosks in malls, residential areas, and metro stations. "
            "Understanding these preferred zones helps optimize deployment and ensures maximum user convenience."
        )
        fig, ax = plt.subplots(figsize=(5,3))
        loc_counts = df["Refill_Location"].value_counts()
        sns.barplot(x=loc_counts.index, y=loc_counts.values, ax=ax)
        ax.set_title("Preferred Locations for ReFill Hub")
        plt.xticks(rotation=45)
        st.pyplot(fig)
