# =============================================================================
# Copyright (c) 2023, ynjassionchen@gmail.com
# =============================================================================
cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(${CMAKE_PROJECT_NAME}Tests VERSION 1.0.0 LANGUAGES CXX)

enable_testing()
file(GLOB TEST_SOURCES "*.cpp")
find_package(GTest REQUIRED)


foreach(file ${TEST_SOURCES})
    # Define each unitest executable
    string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.cpp)" "\\2" test_name ${file}) 
    add_executable(${test_name}_Tests ${file})
    
    include(GoogleTest)
    target_link_libraries(
        ${test_name}_Tests
        GTest::gtest_main
        kvikdataset_lib
    )

    get_target_property(OUT ${test_name}_Tests LINK_LIBRARIES)
    add_test(
        NAME
            ${test_name}
        COMMAND
            ${test_name}_Tests        
    )
endforeach()