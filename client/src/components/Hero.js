import React from 'react';
import Typed from 'react-typed';

const Hero = () => {
  return (
    <div className='text-black ml-52'>
      <div className='max-w-[800px] mt-[-60px] w-full h-screen ml-5 text-left flex flex-col justify-center text-opacity-90'>
        <h1 className='md:text-8xl sm:text-7xl text-4xl font-bold md:py-2 mix-blend-overlay'>
          Trading with AI.
        </h1>
        <h1 className='md:text-6xl sm:text-5xl text-4xl font-bold mix-blend-overlay'>
          Built by 
          <Typed
          className='md:text-6xl sm:text-5xl text-4xl font-bold md:pl-3 mix-blend-normal'
            strings={['Experts', 'Scientists', 'Researchers', 'Engineers']}
            typeSpeed={80}
            backSpeed={50}
            backDelay={1500}
            loop
          />
        </h1>

        <h1 className='md:text-4xl sm:text-3xl text-4xl font-bold md:py-1 mix-blend-overlay'>
          Easy for You
        </h1>
        <p className='w-[600px] mt-16 mix-blend-mulitply md:text-xl text-l text-black text-opacity-75'>Join our pool of investors while tracking and splitting profits of our proprietary trading software.</p>
        <button className='bg-black w-[200px] rounded-full font-medium my-1 py-3 text-white transform transition duration-500 hover:scale-110'>Get Started</button>
      </div>
    </div>
  );
};

export default Hero;