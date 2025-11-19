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
from mpl_toolkits.mplot3d import Axes3D   # ⭐ FOR 3D MODEL ⭐

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

    # GREEN BOX
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

        st.write("""
        **MAE** tells how far predictions are from the real willingness-to-pay.  
        **RMSE** penalizes larger errors more heavily.  
        """)

        residuals = y_test - preds
        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(x=preds, y=residuals, alpha=0.6, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        st.pyplot(fig)


    # ⭐⭐⭐ UPDATED: CLUSTERING WITH 3D MODEL ⭐⭐⭐
    with tabs[2]:
        st.header("Customer Clustering")
        k=st.slider("Number of clusters",2,6,3)

        if st.button("Run Clustering"):
            df_num=df.select_dtypes(include=['int64','float64'])

            km=KMeans(n_clusters=k,random_state=42).fit(df_num)
            df['Cluster']=km.labels_
            st.dataframe(df['Cluster'].value_counts())

            # ----- 3D PCA -----
            pca_3=PCA(n_components=3)
            p3=pca_3.fit_transform(df_num)

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(
                p3[:,0], p3[:,1], p3[:,2],
                c=df["Cluster"], cmap='viridis', s=40
            )

            ax.set_title("3D Interactive Customer Clusters (Rotatable)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            fig.colorbar(sc)
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


    # Insights
    with tabs[4]:
        st.header("Insights")

        st.subheader("1. Eco-aware users show higher adoption")
        fig, ax = plt.subplots(figsize=(3.8,2.3))
        sns.barplot(
            x=df["Likely_to_Use_ReFillHub"], 
            y=df["Uses_Eco_Products"],
            palette="viridis",
            estimator=np.mean,
            ax=ax
        )
        st.pyplot(fig)

        st.subheader("2. Mid-income consumers show the strongest adoption")
        fig, ax = plt.subplots(figsize=(3.8,2.3))
        sns.boxplot(x=df["Income"], y=df["Likely_to_Use_ReFillHub"], palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("3. Plastic ban awareness boosts interest")
        awareness=df["Aware_Plastic_Ban"].value_counts()
        fig, ax = plt.subplots(figsize=(3.2,3.2))
        ax.pie(awareness.values, labels=awareness.index, autopct="%1.1f%%")
        st.pyplot(fig)

        st.subheader("4. Sustainability score → Higher WTP")
        fig, ax = plt.subplots(figsize=(3.8,2.3))
        sns.regplot(x=df["Reduce_Waste_Score"], y=df["Willingness_to_Pay_AED"], ax=ax)
        st.pyplot(fig)

        st.subheader("5. Popular refill locations")
        loc=df["Refill_Location"].value_counts()
        fig, ax = plt.subplots(figsize=(4,2.3))
        sns.lineplot(x=loc.index, y=loc.values, marker="o", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
