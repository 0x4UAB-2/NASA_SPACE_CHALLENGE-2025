

//given the new value, the previus data and the time elapsed since the last time, returns the previus data shifted dt amount and adding the new data on the free spots
function getNewYValues(newValue, prevData) {
    const arraySize = 500;
    dt = 1
    for(let i=0; i<arraySize-dt; i++){
        prevData[i] = prevData[i+1]; 
    }
    for(let i=arraySize-1; i>=arraySize-dt; i--){
        prevData[i]=newValue;
    }

    return prevData;
}

// Given r1,r2 (radi of two circumferences) and d (distance between centers), returns the area of set diferenci between the
// first and second circumferneces.
function areaDifference(r1, r2, d) {
  // Case 1: No overlap
  if (d >= r1 + r2) {
    return Math.PI * r1 * r1;
  }

  // Case 2: C1 completely inside C2
  if (d <= r2 - r1) {
    return 0;
  }

  // Case 3: C2 completely inside C1
  if (d <= r1 - r2) {
    return Math.PI * (r1 * r1 - r2 * r2);
  }

  // Case 4: Partial overlap
  const alpha = Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
  const beta  = Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));

  const overlap = r1 * r1 * alpha + r2 * r2 * beta
                - 0.5 * Math.sqrt(
                    (-d + r1 + r2) *
                    (d + r1 - r2) *
                    (d - r1 + r2) *
                    (d + r1 + r2)
                  );

  return Math.PI * r1 * r1 - overlap;
}
















// Functions done by mi pana Gemini


function getTransitFlux(z, r_p) {
    // z: separation of centers in units of stellar radii
    // r_p: planet radius in units of stellar radii
    if (z > 1 + r_p) {
        // No overlap
        return 1.0;
    }
    if (z <= 1 - r_p) {
        // Planet fully within the star
        return 1.0 - r_p**2;
    }
    if (z <= r_p - 1) {
        // Star fully within the planet (annular eclipse) - not a typical transit
        return 0.0;
    }

    // Partial overlap
    const k0 = Math.acos((z**2 + r_p**2 - 1) / (2 * z * r_p));
    const k1 = Math.acos((z**2 + 1 - r_p**2) / (2 * z));
    const area = (r_p**2 * k0) + k1 - 0.5 * Math.sqrt(4*z**2 - (z**2 - r_p**2 + 1)**2);

    return 1.0 - area / Math.PI;
}


function calculateKoiFeaturesJS(starRadius, starMass, starTemp, bodyRadius, orbitalPeriod, impactParam) {
    const koi_period = orbitalPeriod;
    
    const koi_ror = (bodyRadius / 109.2) / starRadius;


    const P_years = orbitalPeriod / 365.25;
    const M_solar = starMass;
    const koi_sma = (P_years**2 * M_solar)**(1/3);

    const koi_teq = starTemp * Math.sqrt(starRadius / (2 * koi_sma));

    return {
        "koi_period": koi_period,
        "koi_ror": koi_ror,
        "koi_ingress": 0, // Simplified model doesn't calculate ingress
        "koi_impact": impactParam,
        "koi_prad": bodyRadius,
        "koi_sma": koi_sma,
        "koi_teq": koi_teq,
    };
}