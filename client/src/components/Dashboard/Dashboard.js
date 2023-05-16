import React, { useContext, useEffect, useState } from "react";
import ThemeContext from "../../context/ThemeContext";
import Overview from "./Overview";
import Details from "./Details";
import Chart from "../Chart";
import Header from "./Header";
import Sidebar from "./Sidebar";
import StockContext from "../../context/StockContext";
import { fetchStockDetails, fetchQuote } from "../../utils/api/stock-api";

const Dashboard = () => {
  const { darkMode } = useContext(ThemeContext);

  const { stockSymbol } = useContext(StockContext);

  const [stockDetails, setStockDetails] = useState({});

  const [quote, setQuote] = useState({});

  useEffect(() => {
    const updateStockDetails = async () => {
      try {
        const result = await fetchStockDetails(stockSymbol);
        setStockDetails(result);
      } catch (error) {
        setStockDetails({});
        console.log(error);
      }
    };

    const updateStockOverview = async () => {
      try {
        const result = await fetchQuote(stockSymbol);
        setQuote(result);
      } catch (error) {
        setQuote({});
        console.log(error);
      }
    };

    updateStockDetails();
    updateStockOverview();
  }, [stockSymbol]);

  return (
    <div
      className={`h-screen grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 grid-rows-8 md:grid-rows-7 xl:grid-rows-5 auto-rows-fr gap-6 p-10 font-quicksand ${
        darkMode ? "bg-gray-900 text-gray-300" : "bg-neutral-100"
      }`}
    >
      <div className="hidden md:block md:col-span-1 xl:col-span-2 xl:row-start-1 xl:row-span-5">
        <Sidebar />
      </div>
      <div className="col-span-1 md:col-span-2 xl:col-span-1 row-span-1">
        <div className="flex flex-col md:flex-row items-start md:items-center">
          <Header name={stockDetails.name} />
          <div className="mt-4 md:mt-0 md:ml-4">
            <Overview
              symbol={stockSymbol}
              price={quote.pc}
              change={quote.d}
              changePercent={quote.dp}
              currency={stockDetails.currency}
            />
          </div>
        </div>
      </div>
      <div className="md:col-span-2 xl:col-start-2 xl:col-span-2 row-span-4">
        <Chart />
      </div>
      <div className="row-span-2 xl:row-span-3 flex items-end">
        <Details details={stockDetails} />
      </div>
    </div>
  );
  
  
  
};

export default Dashboard;