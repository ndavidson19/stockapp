import React from 'react';
import Single from '../assets/single.png'
import Double from '../assets/double.png'
import Triple from '../assets/triple.png'
import BeginAlien from '../assets/beginalien.png'
import InterAlien from '../assets/interalien.png'
import AdvAlien from '../assets/advancedalien.png'
import SignIn from './SignIn';
import  { useNavigate } from 'react-router-dom';



const Cards = (props) => {

    const navigate = useNavigate();
    const handleClick = () => {
        navigate('/signin')
    }


  return (
    <div ref = {props.cardsRef} id="section1" className='w-full py-[10rem] px-4 bg-white'>
      <div className='max-w-[1240px] mx-auto grid md:grid-cols-3 gap-8 delay-200'>
          <div className='w-full shadow-xl flex flex-col p-4 my-4 rounded-lg hover:scale-105 duration-300'>
              <img className='w-20 mx-auto mt-[-3rem] bg-white' src={BeginAlien} alt="/" />
              <h2 className='text-2xl font-bold text-center py-8'>Beginner Trader</h2>
              <p className='text-center text-4xl font-bold'>$500 +</p>
              <div className='text-center font-medium'>
                  <p className='py-2 border-b mx-8 mt-8'>Profit Sharing</p>
                  <p className='py-2 border-b mx-8'>Automated Trades</p>
                  <p className='py-2 border-b mx-8'>Send up to 2 GB</p>
              </div>
              <button className='bg-[#00df9a] w-[200px] rounded-md font-medium my-6 mx-auto px-6 py-3'>Start Trial</button>
          </div>
          <div className='w-full shadow-xl bg-gray-100 flex flex-col p-4 md:my-0 my-8 rounded-lg hover:scale-105 duration-300'>
              <img className='w-20 mx-auto mt-[-3rem] bg-transparent' src={InterAlien} alt="/" />
              <h2 className='text-2xl font-bold text-center py-8'>Intermediate Trader</h2>
              <p className='text-center text-4xl font-bold'>$10,000 +</p>
              <div className='text-center font-medium'>
                  <p className='py-2 border-b mx-8 mt-8'>500 GB Storage</p>
                  <p className='py-2 border-b mx-8'>1 Granted User</p>
                  <p className='py-2 border-b mx-8'>Send up to 2 GB</p>
              </div>
              <button className='bg-black text-[#00df9a] w-[200px] rounded-md font-medium my-6 mx-auto px-6 py-3'>Start Trial</button>
          </div>
          <div className='w-full shadow-xl flex flex-col p-4 my-4 rounded-lg hover:scale-105 duration-300'>
              <img className='w-20 mx-auto mt-[-3rem] bg-white' src={AdvAlien} alt="/" />
              <h2 className='text-2xl font-bold text-center py-8'>Advanced Trader</h2>
              <p className='text-center text-4xl font-bold'>$100,000 +</p>
              <div className='text-center font-medium'>
                  <p className='py-2 border-b mx-8 mt-8'>500 GB Storage</p>
                  <p className='py-2 border-b mx-8'>1 Granted User</p>
                  <p className='py-2 border-b mx-8'>Send up to 2 GB</p>
              </div>
              <button onClick = {handleClick} className='bg-[#00df9a] w-[200px] rounded-md font-medium my-6 mx-auto px-6 py-3'>Start Trial</button>
          </div>
      </div>
    </div>
  );
};

export default Cards;