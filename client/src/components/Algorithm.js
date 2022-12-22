import React from 'react';
import LottieControl from './RobotAnimation';


const Algorithm = () => {
  return (
    <div className='w-full bg-black py-16 px-4'>
      <div className='max-w-[1240px] mx-auto grid md:grid-cols-2'>
         <div className='flex flex-col justify-center'>
          <p className='text-[#00df9a] font-bold '>ARTIFICIAL INTELLIGENCE MODELS</p>
          <h1 className='text-white md:text-4xl sm:text-3xl text-2xl font-bold py-2'>Decisions Powered by AI Algorithms</h1>
          <p className='text-white'>
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Voluptatum
            molestiae delectus culpa hic assumenda, voluptate reprehenderit
            dolore autem cum ullam sed odit perspiciatis. Doloribus quos velit,
            eveniet ex deserunt fuga?
          </p>
          <button className='bg-[#00df9a] text-black w-[200px] rounded-md font-medium my-6 mx-auto md:mx-0 py-3'>Learn More</button>
        </div>
        <LottieControl />
      </div>
    </div>
  );
};

export default Algorithm;