# ExoFinder

**Summary**  
ExoFinder is a web application that allows users, scientists, and enthusiasts to explore new exoplanet datasets through interactive visualizations of exoplanets detected by machine learning models. It is designed to be user-friendly while including advanced features for more experienced users.

**Project Demonstration**  
[Demo Video](https://uab-my.sharepoint.com/:b:/g/personal/1666599_uab_cat/EboaFGqA9qlGqMImbE3pEZUBfxQfT-acZaVm0mrCXpjg-A?e=xXaBjB)

**Project Repository**  
[GitHub](https://github.com/0x4UAB-2/NASA_SPACE_CHALLENGE-2025)

## Project Details

ExoFinder simulates the **Transit Method** used in astrophysics to detect exoplanets via machine learning. The application allows users to adjust stellar and planetary parameters—such as radius, mass, orbital period, and temperature—and observe how an exoplanet transits across its star in real time. During the transit, the system calculates the drop in observed flux and generates a dynamic **light curve** using Chart.js.

The simulation also extracts scientifically relevant features, including:
- Impact parameter
- Planet-to-star radius ratio
- Equilibrium temperature  

These features are displayed in a **KOI (Kepler Object of Interest) style** and can be used for ML classification in the second phase using models like **Random Forest, CatBoost, XGBoost**, and ensemble methods.

The frontend is developed with **HTML, CSS, and JavaScript**, following a modular structure separating UI handling and astrophysical calculations. A key interactive feature is that the exoplanet appears only when the user hovers over the star, providing an intuitive understanding of transit geometry. Noise can also be toggled on/off to simulate real telescope observations.

**Use Cases**
- Educational: understand exoplanetary transits and light curves.
- Research prototyping: simulate features for machine learning applications in exoplanet detection.

## Use of Artificial Intelligence

Limited AI tools were used to accelerate development efficiency:
- AI code assistants (e.g., ChatGPT/Gemini) helped generate small UI interaction snippets and frontend logic.
- No AI-generated images, videos, audio, or datasets were used; all media elements are copyright-free.
- All scientific calculations and simulations were manually implemented and verified.  

All AI-generated code suggestions were reviewed, adapted, and validated by the team to ensure correctness and originality.

## Technologies Used
- **Frontend:** HTML, CSS, JavaScript, Chart.js  
- **Machine Learning:** Python, Random Forest, CatBoost, XGBoost  
- **Visualization:** Interactive light curves, KOI-style metrics  

---

