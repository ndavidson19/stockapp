import React from 'react'
import ThemeIcon from '../ThemeIcon';
import { GiAbstract005, GiAbstract037, GiAlienBug, GiAmethyst, GiAndroidMask } from "react-icons/gi";


const Sidebar = () => {

  return (
    <div className="fixed top-0 left-0 h-screen w-16 m-0 flex flex-col bg-gray-900 text-white shadow-lg">
        <SideBarIcon icon={<GiAbstract005 size = '28'/>} text='Home' />
        <SideBarIcon icon={<GiAbstract037 size = '28'/>} text='Explore' />
        <SideBarIcon icon={<GiAlienBug size = '28'/>} text='Notifications' />
        <SideBarIcon icon={<GiAmethyst size = '28'/>} text='Messages' />
        <SideBarIcon icon={<GiAndroidMask size = '28'/>} text='Bookmarks' />
        <SideBarIcon icon={<ThemeIcon size = '28'/>} text='Dark Mode' />
    </div>
  )
}

const SideBarIcon = ({ icon, text = 'tooltip ' }) => (
        <div className="sidebar-icon group">
            {icon}
            <span className='sidebar-tooltip group-hover:scale-100'>
                {text}
            </span>
        </div>
)


export default Sidebar 