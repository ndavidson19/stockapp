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
import SignIn from "./components/SignIn";
import { BrowserRouter, Route, useNavigate, Routes,  } from "react-router-dom";
import LandingPage from "./components/LandingPage";

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [stockSymbol, setStockSymbol] = useState("SPY");
  const [loggedIn, setLoggedIn] = useState(true);
  const cardsRef = useRef(null);

  const scrollToCards = () => {
    cardsRef.current.scrollIntoView({ behavior: "smooth" });
  };


    // if not logged in display hero
    // if logged in display dashboard
  


    if (loggedIn) {
      return(
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<LandingPage/>}> 
              <Route path="/signin" element={<SignIn/>}/>
            </Route>
          </Routes>
        </BrowserRouter>
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