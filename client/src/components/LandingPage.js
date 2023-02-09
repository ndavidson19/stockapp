import { useState, useRef } from "react";
import Analytics from "./Analytics";
import Dashboard from "./Dashboard";
import Hero from "./Hero";
import Algorithm from "./Algorithm";
import Navbar from "./Navbar";
import Cards from "./Cards";
import Newsletter from "./Newsletter";
import StockContext from "../context/StockContext";
import ThemeContext from "../context/ThemeContext";
import AnimatedNavbar from "./AnimatedNavbar";


/*

          <div> 
            <AnimatedNavbar duration={ 300 } />
            <Hero refs={cardsRef} scrollToCards = {scrollToCards}/>
            <Analytics />
            <Algorithm />
            <Cards ref = {cardsRef}/>
            <Newsletter />
          </div>


*/


function LandingPage() {
  const [darkMode, setDarkMode] = useState(false);
  const [stockSymbol, setStockSymbol] = useState("SPY");
  const [loggedIn, setLoggedIn] = useState(true);
  const cardsRef = useRef(null);

  const scrollToCards = () => {
    cardsRef.current.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <ThemeContext.Provider value={{ darkMode, setDarkMode }}>
      <StockContext.Provider value={{ stockSymbol, setStockSymbol }}>
        <div>
          <AnimatedNavbar />
          <Hero refs={cardsRef} scrollToCards={scrollToCards} />
          <Analytics />
          <Algorithm />
          <Cards ref={cardsRef} />
          <Newsletter />
        </div>
      </StockContext.Provider>
    </ThemeContext.Provider>
  );
}


export default LandingPage;