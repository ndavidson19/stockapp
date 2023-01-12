import { useState, useRef } from "react";
import "./App.css";
import Analytics from "./components/Analytics";
import Dashboard from "./components/Dashboard";
import Hero from "./components/Hero";
import Algorithm from "./components/Algorithm";
import Navbar from "./components/Navbar";
import Cards from "./components/Cards";
import Newsletter from "./components/Newsletter";
import StockContext from "./context/StockContext";
import ThemeContext from "./context/ThemeContext";
import AnimatedNavbar from "./components/AnimatedNavbar";



function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [stockSymbol, setStockSymbol] = useState("SPY");
  const [loggedIn, setLoggedIn] = useState(false);
  const cardsRef = useRef(null);

  const scrollToCards = () => {
    cardsRef.current.scrollIntoView({ behavior: "smooth" });
  };


    // if not logged in display hero
    // if logged in display dashboard
  


    if (!loggedIn) {
      return(
      <div>
        <AnimatedNavbar duration={ 300 } />
        <Hero refs={cardsRef} scrollToCards = {scrollToCards}/>
        <Analytics />
        <Algorithm />
        <Cards ref = {cardsRef}/>
        <Newsletter />
      </div>

      )
    }
    else {
    return (
      <ThemeContext.Provider value={{ darkMode, setDarkMode }}>
        <StockContext.Provider value={{ stockSymbol, setStockSymbol }}>
          <Dashboard />
        </StockContext.Provider>
      </ThemeContext.Provider>
    );
    }
    
}

export default App;