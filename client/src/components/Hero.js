

import React, { useRef } from 'react';
import Typed from 'react-typed';
import { Canvas } from '@react-three/fiber'
import Rocket from './Rocket';



const Hero = (props) => {

  const ScrollToSection1 = props.scrollToCards;

  return (
    <div className='text-black ml-52 flex'>
      <div className='max-w-[900px] mt-[-50px] w-full h-screen ml-5 text-left flex flex-col justify-center text-opacity-80'>
        <h1 className='md:text-8xl sm:text-7xl text-4xl flex font-bold md:py-2 mix-blend-overlay'>
          Trade with AI.
        </h1>
        <h1 className='md:text-6xl sm:text-5xl text-4xl flex font-bold mix-blend-overlay'>
          Built by 
          <Typed
          className='md:text-6xl sm:text-5xl flex text-4xl font-bold md:pl-3 mix-blend-normal'
            strings={['Experts', 'Scientists', 'Researchers', 'Engineers']}
            typeSpeed={80}
            backSpeed={50}
            backDelay={1500}
            loop
          />
        </h1>

        <h1 className='md:text-4xl sm:text-3xl text-2xl flex font-bold md:py-2 mix-blend-overlay'>
          Easy for You
        </h1>
        <p className='w-[600px] mt-16 mix-blend-mulitply md:text-xl flex text-l text-black text-opacity-75'>Join our pool of investors while tracking and splitting profits of our proprietary trading software.</p>
        <button type='button' onClick={ScrollToSection1} className='bg-black w-[200px] rounded-full font-medium my-1 py-3 text-white transform transition duration-500 hover:scale-110'>Get Started</button>
      </div>
      <div className='w-full h-screen'>
        <Canvas>
          <Rocket />
        </Canvas>
    </div>
    </div>
  );
};



export default Hero;
