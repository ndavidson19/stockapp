/*
  Auto-generated by Spline
*/
import React, { useRef, useState } from 'react'
import useSpline from '@splinetool/r3f-spline'
import { OrthographicCamera } from '@react-three/drei'
import { OrbitControls } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'


export default function Rocket({ ...props }) {
  const { nodes, materials } = useSpline('https://prod.spline.design/DlqMvCgSI6Kbihru/scene.splinecode')
  const rocketRef = useRef()

  // Have the rocket position oscillate up and down
  const [rocketPosition, setRocketPosition] = useState(0)


  useFrame((state, delta) => {
    rocketRef.current.rotation.y += 0.007555
    const elapsedTime = state.clock.getElapsedTime()
    setRocketPosition(Math.sin(elapsedTime) * 200)
  })

  return (
    <>
        <OrbitControls   minPolarAngle={Math.PI / 6}
  maxPolarAngle={Math.PI - Math.PI / 6} />
      <group {...props} dispose={null}>
        <OrthographicCamera 
          makeDefault={true}
          zoom={1}
          far={100000}
          near={-100000}
          position={[125, 200, rocketPosition + 450]}
          rotation={[0.13, 0.1, -0.01]}
        />
        <group name="ROCKET" position={[0, 130, 0]} ref={rocketRef}>
          <pointLight 
            name="Point Light 2"
            intensity={1}
            distance={5387}
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
            shadow-camera-near={100}
            shadow-camera-far={2500}
            color="#fed9c1"
            position={[0, -316.4, -225.24]}
          />
          <pointLight 
            name="Point Light"
            intensity={1}
            distance={5387}
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
            shadow-camera-near={100}
            shadow-camera-far={2500}
            color="#fed9c1"
            position={[0, -316.4, 113.79]}
          />
          <mesh 
            name="Blur effect"
            geometry={nodes['Blur effect'].geometry}
            material={materials['Blur effect Material']}
            castShadow
            receiveShadow
            position={[1.75, -565.68, 1.01]}
            scale={[3, 7, 3]}
          />
          <mesh 
            name="Fire"
            geometry={nodes.Fire.geometry}
            material={materials['Fire Material']}
            castShadow
            receiveShadow
            position={[1.11, -509.15, 1.01]}
            scale={[2, 5.39, 1.97]}
          />
          <mesh  
            name="Cap"
            geometry={nodes.Cap.geometry}
            material={materials['Cap Material']}
            castShadow
            receiveShadow
            position={[1.18, 89.23, 34.17]}
            scale={[0.86, 1, 1]}
          />
          <mesh 
            name="Fins"
            geometry={nodes.Fins.geometry}
            material={materials['Fins Material']}
            castShadow
            receiveShadow
            position={[4.67, -97.33, 0]}
            rotation={[0, -0.71, 0]}
            scale={1}
          />
          <mesh 
            name="Bottom2"
            geometry={nodes.Bottom2.geometry}
            material={materials['Bottom2 Material']}
            castShadow
            receiveShadow
            position={[-0.37, -250.65, 1.08]}
            rotation={[Math.PI, Math.PI / 2, 0]}
            scale={1}
          />
          <mesh 
            name="Cylinder" 
            geometry={nodes.Cylinder.geometry}
            material={materials['Cylinder Material']}
            castShadow
            receiveShadow
            position={[-0.37, -234.39, 1.08]}
            rotation={[Math.PI, Math.PI / 2, 0]}
            scale={1}
          />
          <mesh 
            name="BODY"
            geometry={nodes.BODY.geometry}
            material={materials['BODY Material']}
            castShadow
            receiveShadow
            rotation={[-Math.PI, -0.09, -Math.PI]}
            scale={5}
          />
        </group>
        <directionalLight
          name="Directional Light"
          castShadow
          intensity={1}
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
          shadow-camera-near={-10000}
          shadow-camera-far={100000}
          shadow-camera-left={-362.778}
          shadow-camera-right={362.778}
          shadow-camera-top={362.778}
          shadow-camera-bottom={-362.778}
          position={[451.99, 300, 110.42]}
        />
        <hemisphereLight name="Default Ambient Light" intensity={0.75} color="#eaeaea" />
      </group>
    </>
  )
}
