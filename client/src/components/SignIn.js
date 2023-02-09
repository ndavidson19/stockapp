import React from "react";


/// This will render the sign in page
// * @param {object} props
// * @param {function} props.onSignIn - Callback function to be called when the user signs in
// * @param {string} props.username - The username of the user
// * @param {string} props.password - The password of the user
// * @param {string} props.error - The error message to display
// * @param {boolean} props.loading - Whether the sign in form is loading
// * @param {function} props.onUsernameChange - Callback function to be called when the username input changes
// * @param {function} props.onPasswordChange - Callback function to be called when the password input changes
// * @param {function} props.onSignIn - Callback function to be called when the user signs in
// * @param {function} props.onSignUp - Callback function to be called when the user signs up
// * @param {function} props.onForgotPassword - Callback function to be called when the user clicks the forgot password link

const SignIn = ({
    username,
    password,
    error,
    loading,
    onUsernameChange,
    onPasswordChange,
    onSignIn,
    onSignUp,
    onForgotPassword,
}) => {
    return (
        <div className="flex flex-col items-center justify-center h-screen">
            <div className="w-full max-w-xs">
                <form className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
                    <div className="mb-4">
                        <label
                            className="block text-gray-700 text-sm font-bold mb-2"
                            htmlFor="username"
                        >
                            Username
                        </label>
                        <input
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            id="username"
                            type="text"
                            placeholder="Username"
                            value={username}
                            onChange={onUsernameChange}
                        />
                    </div>
                    <div className="mb-6">
                        <label
                            className="block text-gray-700 text-sm font-bold mb-2"
                            htmlFor="password"
                        >
                            Password
                        </label>
                        <input
                            className="shadow appearance-none border border-red-500 rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline"
                            id="password"
                            type="password"
                            placeholder="******************"
                            value={password}
                            onChange={onPasswordChange}
                        />
                        <p className="text-red-500 text-xs italic">
                            Please choose a password.
                        </p>
                    </div>
                    <div className="flex items-center justify-between">
                        <button
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                            type="button"
                            onClick={onSignIn}
                        >
                            Sign In
                        </button>
                        <a
                            className="inline-block align-baseline font-bold text-sm text-blue-500 hover:text-blue-800"
                            href="#"
                            onClick={onForgotPassword}
                        >
                            Forgot Password?
                        </a>
                    </div>
                </form>
                <p className="text-center text-gray-500 text-xs">
                    &copy;2020 Acme Corp. All rights reserved.
                </p>
            </div>
        </div>
    );
}

export default SignIn;
