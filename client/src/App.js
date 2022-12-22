import { useState } from "react";
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

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [stockSymbol, setStockSymbol] = useState("SPY");
  const [loggedIn, setLoggedIn] = useState(false);
    // if not logged in display hero
    // if logged in display dashboard
    if (!loggedIn) {
      return(
      <div>
        <Navbar />
        <Hero/>
        <Analytics />
        <Algorithm />
        <Cards />
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